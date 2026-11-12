/** Routes: where a model call goes and how it authenticates. */

import { YianTypeError } from "../typesys/errors.ts";
import { type Auth, checkConnectionHeaderValue } from "./auth.ts";

// An RFC 7230 token.
const HEADER_NAME = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
// The openai and anthropic SDKs send this Accept on every request, streamed ones too.
const DEFAULT_HEADERS = { "content-type": "application/json", accept: "application/json" };

/** One provider endpoint and how to call it. */
export class Route {
  /** The provider name, used in error messages. */
  readonly provider: string;
  /** The API base, for example `"https://api.openai.com/v1"`; a trailing slash is allowed. */
  readonly baseUrl: string;
  /**
   * The endpoint relative to `baseUrl`, for example `"responses"` or `"chat/completions"`; a
   * leading slash is allowed.
   */
  readonly path: string;
  /** The credential headers, resolved once per attempt. */
  readonly auth: Auth;
  /** Query parameters, for example `{"api-version": "..."}`; a frozen copy. */
  readonly query: Readonly<Record<string, string>>;
  /**
   * Non-secret extra headers; a frozen copy. They replace the framing defaults (`Content-Type`
   * and `Accept`) of the same name, case-insensitively; auth headers of the same name replace
   * them.
   */
  readonly headers: Readonly<Record<string, string>>;

  /**
   * @throws TypeError If a header name is not an HTTP token or `checkConnectionHeaderValue`
   *   refuses a value; the message names the header, never the value.
   */
  constructor(options: {
    provider: string;
    baseUrl: string;
    path: string;
    auth: Auth;
    query?: Record<string, string>;
    headers?: Record<string, string>;
  }) {
    this.provider = options.provider;
    this.baseUrl = options.baseUrl;
    this.path = options.path;
    this.auth = options.auth;
    // Copied so a caller mutating its own object cannot change a route in use.
    this.query = Object.freeze(Object.fromEntries(Object.entries(options.query ?? {})));
    this.headers = Object.freeze(Object.fromEntries(Object.entries(options.headers ?? {})));
    // Checked here because both runtimes echo an invalid header value in their own error,
    // and host header values may be secrets.
    for (const [name, value] of Object.entries(this.headers)) {
      if (!HEADER_NAME.test(name)) {
        throw new YianTypeError(`route header name ${JSON.stringify(name)} is not a valid HTTP header name`);
      }
      checkConnectionHeaderValue(name, value);
    }
  }

  /**
   * Resolve the auth and return one request's headers: `content-type` and `accept` of
   * `application/json`, replaced by this route's headers, then by the auth headers, of any case.
   *
   * @throws What the auth source throws, unchanged.
   */
  async requestHeaders(): Promise<Headers> {
    const headers = new Headers(DEFAULT_HEADERS);
    for (const [name, value] of Object.entries(this.headers)) headers.set(name, value);
    for (const [name, value] of Object.entries(await this.auth.headers())) headers.set(name, value);
    return headers;
  }

  /**
   * Join `baseUrl` and `path` with exactly one slash, then add the encoded query, if any.
   *
   * @returns The endpoint URL.
   */
  url(): string {
    const url = `${this.baseUrl.replace(/\/+$/, "")}/${this.path.replace(/^\/+/, "")}`;
    const query = new URLSearchParams(this.query).toString();
    return query ? `${url}?${query}` : url;
  }
}
