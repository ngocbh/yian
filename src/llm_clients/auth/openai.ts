/** OpenAI browser/device OAuth, refresh, and authenticated model discovery. */

import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import net from "node:net";

import { current } from "../../defaults.ts";
import { compareCodeUnits } from "../../diagnostics.ts";
import { realScheduler, type Scheduler, USER_AGENT } from "../../platform.ts";
import { isRecord } from "../../typesys/index.ts";
import { openaiIdHeaders, validateHeaders } from "../connection_settings.ts";
import { OPENAI_API_ENDPOINT, SUBSCRIPTION_ENDPOINT } from "../endpoints.ts";
import { type ApiKeyCredentials, AuthError, HttpResponse, isVisibleText, OAuthCredentials } from "./common.ts";
import { request as httpRequest } from "./http.ts";

const CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const ISSUER = "https://auth.openai.com";
const REDIRECT_URI = "http://localhost:1455/auth/callback";

/** A decimal float: optional sign, digits with single `_` between them, an optional fraction and exponent. */
const FLOAT_TEXT = /^\s*[+-]?(\d(_?\d)*(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?\s*$/;

/** The options `OpenAIAuth` passes to its `request`. */
export interface AuthRequestOptions {
  headers: Record<string, string>;
  timeoutMs: number;
  jsonBody?: Readonly<Record<string, unknown>>;
  form?: Readonly<Record<string, string>>;
  signal?: AbortSignal;
}

/** Send one authentication request and return its decoded result, as `http.ts`'s `request` does. */
export type AuthRequest = (method: string, url: string, options: AuthRequestOptions) => Promise<HttpResponse>;

/** A started callback listener: the first callback URL it received, and its close. */
interface CallbackListener {
  readonly callbackUrl: Promise<string>;
  close(): Promise<void>;
}

/** Return the query parameters of a callback URL that starts with `REDIRECT_URI`, ignoring any fragment. */
function callbackParams(url: string): URLSearchParams {
  const rest = url.slice(REDIRECT_URI.length).split("#", 1)[0]!;
  // URLSearchParams drops one leading "?", which parse_qs keeps as part of the first key.
  return new URLSearchParams("&" + rest.slice(1));
}

/** Whether `params` hold exactly one `state`, equal to `state`. */
function stateMatches(params: URLSearchParams, state: string): boolean {
  const states = params.getAll("state");
  return states.length === 1 && sameText(states[0]!, state);
}

/**
 * Return the status, body and, for a valid callback, the callback URL answering `received`.
 *
 * A callback is valid only when it carries this login's `state`: a stale tab or another local
 * process gets a 400, and the login keeps waiting.
 */
function callbackAnswer(
  received: Buffer,
  limit: number,
  state: string,
): { status: string; body: string; target?: string } {
  const bad = { status: "400 Bad Request", body: "Invalid callback request." };
  const lineEnd = received.indexOf("\r\n");
  const line = lineEnd === -1 ? received : received.subarray(0, lineEnd);
  // Checked on bytes: TextDecoder's "ascii" is windows-1252 and Buffer's masks the high bit.
  if (line.some((byte) => byte >= 0x80)) return bad;
  const parts = line.toString("latin1").split(" ");
  if (parts.length !== 3) return bad;
  const [method, target, version] = parts as [string, string, string];
  if (method !== "GET" || (version !== "HTTP/1.0" && version !== "HTTP/1.1")) return bad;
  if (!received.includes("\r\n\r\n") || received.length > limit) return bad;
  if (!/^\/auth\/callback(?:$|[?#])/.test(target)) return { status: "404 Not Found", body: "Unknown callback path." };
  const url = "http://localhost:1455" + target;
  if (!stateMatches(callbackParams(url), state)) {
    return { status: "400 Bad Request", body: "Callback state did not match this login." };
  }
  return { status: "200 OK", body: "Authorization received. Return to Yian.", target: url };
}

/**
 * Start the local login callback listener on 127.0.0.1:1455.
 *
 * Each connection is answered once, after its request headers, the size cap, or the peer's end;
 * `callbackUrl` resolves with the first valid callback carrying `state` and never rejects.
 * Rejects with `AuthError` if the port can't be bound.
 */
async function listen(state: string): Promise<CallbackListener> {
  let found!: (url: string) => void;
  // Created now, so a callback arriving while the browser still opens isn't lost.
  const callbackUrl = new Promise<string>((resolve) => (found = resolve));
  const waiting = new Set<net.Socket>();
  // Half-open, so a peer that ends its side early keeps its socket writable until it is answered,
  // rather than relying on the answer being written before the runtime's automatic end.
  const server = net.createServer({ allowHalfOpen: true }, (socket) => {
    const limit = current().maxOauthCallbackBytes;
    let received = Buffer.alloc(0);
    let answered = false;
    waiting.add(socket);
    const answer = () => {
      if (answered) return;
      answered = true;
      waiting.delete(socket);
      const { status, body, target } = callbackAnswer(received, limit, state);
      if (target !== undefined) found(target);
      const head =
        `HTTP/1.1 ${status}\r\nContent-Type: text/plain; charset=utf-8\r\n` +
        `Content-Length: ${Buffer.byteLength(body)}\r\nConnection: close\r\n\r\n`;
      socket.end(Buffer.from(head + body), () => socket.destroy());
    };
    socket.on("data", (chunk: Buffer) => {
      if (answered) return;
      received = Buffer.concat([received, chunk]);
      if (received.includes("\r\n\r\n") || received.length > limit) answer();
    });
    socket.on("end", answer);
    socket.on("error", answer);
  });
  await new Promise<void>((resolve, reject) => {
    const failed = () => reject(new AuthError("Could not open the local login callback; use device login"));
    server.once("error", failed);
    server.listen({ host: "127.0.0.1", port: 1455 }, () => {
      server.off("error", failed);
      resolve();
    });
  });
  // A later accept failure (EMFILE or the like) must not crash the process.
  server.on("error", () => {});
  return {
    callbackUrl,
    async close() {
      for (const socket of waiting) socket.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}

/**
 * Settle as `work` does, or reject with `signal.reason` once `signal` aborts (at once if it
 * already has), so a host callback or request that never settles can't outlive a stop.
 */
function untilStopped<T>(work: T | PromiseLike<T>, signal: AbortSignal): Promise<T> {
  const settled = Promise.resolve(work);
  // A rejection after the stop has no listener left; keep it from going unhandled.
  settled.catch(() => {});
  return new Promise<T>((resolve, reject) => {
    const stop = () => reject(signal.reason);
    if (signal.aborted) return stop();
    signal.addEventListener("abort", stop, { once: true });
    settled.then(resolve, reject).finally(() => signal.removeEventListener("abort", stop));
  });
}

/** Return `record[key]` if it is an own key, else `fallback`. */
function get(record: Record<string, unknown>, key: string, fallback: unknown = null): unknown {
  return Object.hasOwn(record, key) ? record[key] : fallback;
}

/** Whether a decoded JSON value is set: not null, `false`, zero, `""`, or an empty list or record. */
function truthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  return !(value === null || value === undefined || value === false || value === 0 || value === 0n || value === "");
}

/** Whether two strings have equal UTF-8 bytes, compared in constant time. */
function sameText(a: string, b: string): boolean {
  const left = Buffer.from(a);
  const right = Buffer.from(b);
  return left.length === right.length && timingSafeEqual(left, right);
}

/** Return a 2xx response's body record, or throw `AuthError`. */
function responseData(response: HttpResponse): Record<string, unknown> {
  if (!(response.status >= 200 && response.status < 300)) {
    throw new AuthError(`OpenAI authentication request failed (HTTP ${response.status})`, {
      status: response.status,
    });
  }
  const { data } = response;
  if (!isRecord(data)) throw new AuthError("OpenAI authentication returned malformed data");
  return data;
}

/** Read routing metadata only from a token just received from OpenAI. */
function claimsOf(token: unknown): Record<string, unknown> {
  if (typeof token !== "string") return {};
  const parts = token.split(".");
  if (parts.length !== 3) return {};
  try {
    const raw = new TextDecoder("utf-8", { fatal: true }).decode(Buffer.from(parts[1]!, "base64url"));
    const value: unknown = JSON.parse(raw);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

/** Return the device polling interval in seconds, at least 1, from the server's value. */
function pollInterval(raw: unknown): number {
  let seconds = NaN;
  if (typeof raw === "bigint" || typeof raw === "number") seconds = Number(raw);
  else if (typeof raw === "string" && FLOAT_TEXT.test(raw)) seconds = Number(raw.replaceAll("_", ""));
  if (!Number.isFinite(seconds)) throw new AuthError("OpenAI returned an invalid device polling interval");
  return Math.max(1, seconds);
}

/**
 * Stage validated credentials without changing saved provider selection.
 *
 * `request` receives the method, URL, and options with headers, a timeout, a signal, and an
 * optional `jsonBody` or `form`, and returns an `HttpResponse`; it defaults to `http.ts`'s
 * `request` on this instance's scheduler. Host callbacks own presentation. Aborting `signal`
 * cancels any waiting phase, rejecting with its reason.
 */
export class OpenAIAuth {
  readonly #request: AuthRequest;
  readonly #scheduler: Scheduler;
  readonly #signal: AbortSignal;
  readonly #requestTimeoutMs: number;

  constructor(
    options: { request?: AuthRequest; scheduler?: Scheduler; signal?: AbortSignal; requestTimeoutMs?: number } = {},
  ) {
    const { requestTimeoutMs = 30_000 } = options;
    if (typeof requestTimeoutMs !== "number" || !Number.isFinite(requestTimeoutMs) || requestTimeoutMs <= 0) {
      throw new RangeError("Invalid authentication request timeout");
    }
    this.#scheduler = options.scheduler ?? realScheduler;
    this.#signal = options.signal ?? new AbortController().signal;
    this.#requestTimeoutMs = requestTimeoutMs;
    this.#request =
      options.request ?? ((method, url, sent) => httpRequest(method, url, { ...sent, scheduler: this.#scheduler }));
  }

  /** Run `flow` on a signal that also aborts with `AuthError(message)` after `timeoutMs`. */
  async #withDeadline<T>(timeoutMs: number, message: string, flow: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (typeof timeoutMs !== "number" || !Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new RangeError("Invalid authentication timeout");
    }
    this.#signal.throwIfAborted();
    const deadline = new AbortController();
    const cancel = new AbortController();
    // The rejection handler matters: cancelling the sleep rejects it, and an unhandled
    // rejection ends a Node process.
    void this.#scheduler.sleep(timeoutMs, cancel.signal).then(
      () => deadline.abort(new AuthError(message)),
      () => {},
    );
    try {
      return await flow(AbortSignal.any([this.#signal, deadline.signal]));
    } finally {
      cancel.abort();
    }
  }

  async #send(
    method: string,
    url: string,
    options: {
      signal: AbortSignal;
      headers?: Record<string, string>;
      jsonBody?: Readonly<Record<string, unknown>>;
      form?: Readonly<Record<string, string>>;
    },
  ): Promise<HttpResponse> {
    const { signal, jsonBody, form } = options;
    signal.throwIfAborted();
    // Every request here, auth included, sends `yian/<version>`.
    const headers: Record<string, string> = { "User-Agent": USER_AGENT, ...options.headers };
    const sent: AuthRequestOptions = { headers, timeoutMs: this.#requestTimeoutMs, signal };
    if (jsonBody !== undefined) {
      headers["Content-Type"] = "application/json";
      sent.jsonBody = jsonBody;
    }
    if (form !== undefined) {
      headers["Content-Type"] = "application/x-www-form-urlencoded";
      sent.form = form;
    }
    let response: unknown;
    try {
      response = await untilStopped(this.#request(method, url, sent), signal);
    } catch {
      if (signal.aborted) throw signal.reason;
      // The transport's error can hold request values, so none of it is kept.
      throw new AuthError("OpenAI authentication request failed");
    }
    signal.throwIfAborted();
    if (
      typeof response !== "object" ||
      response === null ||
      Object.getPrototypeOf(response) !== HttpResponse.prototype
    ) {
      throw new AuthError("OpenAI authentication returned an invalid response");
    }
    const { status } = response as HttpResponse;
    if (typeof status !== "number" || !Number.isInteger(status) || status < 100 || status > 599) {
      throw new AuthError("OpenAI authentication returned an invalid response");
    }
    return response as HttpResponse;
  }

  #tokens(data: Record<string, unknown>, previous?: OAuthCredentials): OAuthCredentials {
    const accountIds = new Set<string>();
    let residency: unknown = previous ? previous.residency : null;
    for (const token of [get(data, "id_token"), get(data, "access_token")]) {
      const claims = claimsOf(token);
      const authClaim = get(claims, "https://api.openai.com/auth", {});
      const auth = isRecord(authClaim) ? authClaim : {};
      const topLevel = get(claims, "chatgpt_account_id");
      let account = truthy(topLevel) ? topLevel : get(auth, "chatgpt_account_id");
      const organizations = get(claims, "organizations");
      if (
        (account === null || account === undefined) &&
        Array.isArray(organizations) &&
        organizations.length > 0 &&
        isRecord(organizations[0])
      ) {
        account = get(organizations[0], "id");
      }
      if (account !== null && account !== undefined) {
        if (!isVisibleText(account)) throw new AuthError("OpenAI returned invalid account metadata");
        accountIds.add(account);
      }
      const key = "chatgpt_compute_residency";
      if (Object.hasOwn(auth, key) || Object.hasOwn(claims, key)) {
        residency = get(auth, key, get(claims, key));
        if (residency === "no_constraint") residency = null;
      }
    }
    if (accountIds.size > 1 || (previous && accountIds.size > 0 && !accountIds.has(previous.accountId))) {
      throw new AuthError("OpenAI refresh changed account identity");
    }
    const accountId = accountIds.size > 0 ? [...accountIds][0] : previous ? previous.accountId : null;
    const rawExpiry = get(data, "expires_in", 3600);
    const expiresIn = typeof rawExpiry === "bigint" || typeof rawExpiry === "number" ? Number(rawExpiry) : NaN;
    if (!(expiresIn > 0 && expiresIn <= Number.MAX_VALUE)) {
      throw new AuthError("OpenAI returned invalid credential expiry");
    }
    return new OAuthCredentials({
      accessToken: get(data, "access_token"),
      refreshToken: get(data, "refresh_token", previous ? previous.refreshToken : null),
      expiresAt: this.#scheduler.now() / 1000 + expiresIn,
      accountId,
      residency,
    });
  }

  async #exchange(
    code: unknown,
    verifier: unknown,
    redirectUri: string,
    signal: AbortSignal,
  ): Promise<OAuthCredentials> {
    if (!isVisibleText(code) || !isVisibleText(verifier)) {
      throw new AuthError("OpenAI returned invalid authorization data");
    }
    const response = await this.#send("POST", ISSUER + "/oauth/token", {
      signal,
      form: {
        grant_type: "authorization_code",
        code,
        code_verifier: verifier,
        client_id: CLIENT_ID,
        redirect_uri: redirectUri,
      },
    });
    return this.#tokens(responseData(response));
  }

  /**
   * Authorize with PKCE and a bounded local callback, returning staged tokens.
   *
   * An injected `awaitCallback` receives the redirect URI and the login's signal, which aborts at
   * the deadline or on a stop; it returns the complete callback URL. Without one, this method
   * owns a localhost listener for the entire authorization. `timeoutMs` defaults to ten minutes.
   */
  async browserLogin(
    openUrl: (url: string) => void | Promise<void>,
    options: {
      awaitCallback?: (redirectUri: string, signal: AbortSignal) => string | Promise<string>;
      timeoutMs?: number;
    } = {},
  ): Promise<OAuthCredentials> {
    const { awaitCallback, timeoutMs = 600_000 } = options;
    return this.#withDeadline(timeoutMs, "OpenAI browser login timed out", async (signal) => {
      const verifier = randomBytes(32).toString("base64url");
      const state = randomBytes(32).toString("base64url");
      const challenge = createHash("sha256").update(verifier).digest("base64url");
      const query = new URLSearchParams({
        response_type: "code",
        client_id: CLIENT_ID,
        redirect_uri: REDIRECT_URI,
        scope: "openid profile email offline_access",
        code_challenge: challenge,
        code_challenge_method: "S256",
        id_token_add_organizations: "true",
        codex_cli_simplified_flow: "true",
        state,
        originator: "yian",
      });
      const url = ISSUER + "/oauth/authorize?" + query.toString();
      // Started before the browser opens, so a fast callback isn't lost.
      const listener = awaitCallback === undefined ? await listen(state) : undefined;
      try {
        signal.throwIfAborted();
        await untilStopped(openUrl(url), signal);
        signal.throwIfAborted();
        const result: unknown = await untilStopped(
          awaitCallback === undefined ? listener!.callbackUrl : awaitCallback(REDIRECT_URI, signal),
          signal,
        );
        signal.throwIfAborted();
        if (typeof result !== "string") throw new AuthError("OpenAI returned an invalid browser callback");
        if (
          !(result === REDIRECT_URI || result.startsWith(REDIRECT_URI + "?") || result.startsWith(REDIRECT_URI + "#"))
        ) {
          throw new AuthError("OpenAI returned an invalid callback location");
        }
        const params = callbackParams(result);
        if (!stateMatches(params, state)) throw new AuthError("OpenAI login state did not match");
        if (params.has("error")) throw new AuthError("OpenAI browser authorization was denied");
        const codes = params.getAll("code");
        if (codes.length !== 1 || !codes[0]) {
          throw new AuthError("OpenAI callback did not include an authorization code");
        }
        return await this.#exchange(codes[0], verifier, REDIRECT_URI, signal);
      } finally {
        await listener?.close();
      }
    });
  }

  /**
   * Authorize headlessly using OpenAI's device-auth polling protocol.
   *
   * `displayCode` receives the verification URL and the user code to show. `timeoutMs` defaults
   * to fifteen minutes.
   */
  async deviceLogin(
    displayCode: (verificationUri: string, userCode: string) => void | Promise<void>,
    options: { timeoutMs?: number } = {},
  ): Promise<OAuthCredentials> {
    const { timeoutMs = 900_000 } = options;
    return this.#withDeadline(timeoutMs, "OpenAI login timed out", async (signal) => {
      const data = responseData(
        await this.#send("POST", ISSUER + "/api/accounts/deviceauth/usercode", {
          signal,
          jsonBody: { client_id: CLIENT_ID },
        }),
      );
      const deviceId = get(data, "device_auth_id");
      const userCode = get(data, "user_code", get(data, "usercode"));
      if (!isVisibleText(deviceId) || !isVisibleText(userCode)) {
        throw new AuthError("OpenAI returned invalid device authorization data");
      }
      const interval = pollInterval(get(data, "interval", 5));
      await untilStopped(displayCode(ISSUER + "/codex/device", userCode), signal);
      for (;;) {
        const response = await this.#send("POST", ISSUER + "/api/accounts/deviceauth/token", {
          signal,
          jsonBody: { device_auth_id: deviceId, user_code: userCode },
        });
        if (response.status !== 403 && response.status !== 404) {
          const result = responseData(response);
          return await this.#exchange(
            get(result, "authorization_code"),
            get(result, "code_verifier"),
            ISSUER + "/deviceauth/callback",
            signal,
          );
        }
        // Capped at the login timeout: a huge interval would be a non-finite sleep, and the
        // deadline ends the wait first anyway.
        await this.#scheduler.sleep(Math.min(interval * 1000, timeoutMs), signal);
      }
    });
  }

  /** Refresh staged tokens while preserving the authorized account. */
  async refresh(credentials: OAuthCredentials): Promise<OAuthCredentials> {
    const response = await this.#send("POST", ISSUER + "/oauth/token", {
      signal: this.#signal,
      form: { grant_type: "refresh_token", refresh_token: credentials.refreshToken, client_id: CLIENT_ID },
    });
    return this.#tokens(responseData(response), credentials);
  }

  /**
   * Validate an API key and return the model identifiers it can list, sorted and without
   * duplicates.
   *
   * The request carries the credential's connection headers and, when given, the organization
   * and project ids as `OpenAI-Organization` and `OpenAI-Project`. Rejects with `AuthError` if a
   * setting is invalid (checked before any request, and the message holds no header value or
   * id), the request fails (a mismatched organization or project is an HTTP status), or the
   * list is invalid or empty.
   */
  async apiModels(
    credentials: ApiKeyCredentials,
    options: { organization?: unknown; project?: unknown } = {},
  ): Promise<readonly string[]> {
    let headers: Record<string, string>;
    try {
      headers = validateHeaders(Object.fromEntries(credentials.headers), "openai");
      Object.assign(headers, openaiIdHeaders(options.organization ?? null, options.project ?? null));
    } catch (error) {
      if (error instanceof RangeError) throw new AuthError(`Invalid OpenAI connection settings: ${error.message}`);
      throw error;
    }
    headers.Authorization = "Bearer " + credentials.apiKey;
    const response = await this.#send("GET", `${OPENAI_API_ENDPOINT}/models`, { signal: this.#signal, headers });
    const data = get(responseData(response), "data");
    if (
      !Array.isArray(data) ||
      data.length === 0 ||
      data.some((item) => !isRecord(item) || !isVisibleText(get(item, "id")))
    ) {
      throw new AuthError("OpenAI returned an invalid or empty model list");
    }
    const ids = new Set(data.map((item: Record<string, unknown>) => item.id as string));
    return Object.freeze([...ids].sort(compareCodeUnits));
  }

  /** List visible subscription models, by priority, using the backend catalog protocol. */
  async subscriptionModels(
    credentials: OAuthCredentials,
    options: { clientVersion?: unknown } = {},
  ): Promise<readonly string[]> {
    const { clientVersion = "0.155.1" } = options;
    if (!isVisibleText(clientVersion)) throw new RangeError("Invalid model catalog client version");
    const headers: Record<string, string> = {
      Authorization: "Bearer " + credentials.accessToken,
      "ChatGPT-Account-Id": credentials.accountId,
      originator: "yian",
    };
    if (credentials.residency !== null) headers["x-openai-internal-codex-residency"] = credentials.residency;
    const query = new URLSearchParams({ client_version: clientVersion }).toString();
    const response = await this.#send("GET", `${SUBSCRIPTION_ENDPOINT}/models?${query}`, {
      signal: this.#signal,
      headers,
    });
    const models = get(responseData(response), "models");
    if (
      !Array.isArray(models) ||
      models.some(
        (item) =>
          !isRecord(item) ||
          !isVisibleText(get(item, "slug")) ||
          !["list", "hide", "none"].includes(get(item, "visibility") as string) ||
          typeof get(item, "priority") !== "bigint",
      )
    ) {
      throw new AuthError("OpenAI returned an invalid subscription model list");
    }
    const visible = (models as { slug: string; visibility: string; priority: bigint }[])
      .filter((item) => item.visibility === "list")
      .sort((a, b) => (a.priority < b.priority ? -1 : a.priority > b.priority ? 1 : 0));
    if (visible.length === 0) throw new AuthError("OpenAI returned no available subscription models");
    return Object.freeze([...new Set(visible.map((item) => item.slug))]);
  }
}
