/** Private credential storage: one saved login per provider, read with path and privacy checks. */

// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { userInfo } from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";

import { current, type Env, processEnv } from "../../defaults.ts";
import { realScheduler, type Scheduler } from "../../platform.ts";
import { isRecord } from "../../typesys/index.ts";
import { parseJson } from "../../typesys/json.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type AuthState,
  type Credentials,
  OAuthCredentials,
  ProviderSelection,
} from "./common.ts";

// Entry keys written only when set, so a login without them is stored as before.
const OPTIONAL_ENTRY_KEYS = new Set(["endpoint", "organization", "project"]);
const ENTRY_KEYS = ["method", "model", "credential_id", "credentials"];
const STORE_KEYS = ["version", "default_provider", "providers"];
const OAUTH_KEYS = ["access_token", "refresh_token", "expires_at", "account_id"];

const REGULAR = "Credential storage must use a regular private path";
const MALFORMED = "Credential storage is malformed; reconnect after repairing it";

/** Saved logins by provider, in file order. */
type Providers = Map<string, [ProviderSelection, Credentials | null]>;

/** The open store directory and the identity its descriptor had when checked. */
interface Directory {
  fd: number;
  dev: bigint;
  ino: bigint;
}

/** Options of an `AuthStore`. */
export interface AuthStoreOptions {
  /** The environment read for the default path and for `~` (`HOME`); default `processEnv`. */
  env?: Env;
  /** The clock for token expiry and the sleep for the lock wait; default `realScheduler`. */
  scheduler?: Scheduler;
  /** Once aborted, store calls throw its reason before touching storage. */
  signal?: AbortSignal;
  /** Longest wait for the store's lock, in milliseconds; finite and above 0 (default 30 000). */
  lockTimeoutMs?: number;
  /** How long before expiry a subscription token is refreshed, in milliseconds; finite, 0 or more (default 60 000). */
  refreshMarginMs?: number;
}

/** Whether `value` has exactly the keys `keys`. */
function hasKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

/** Return the storage access error; it has no cause, so no path or system detail shows. */
function accessError(): AuthError {
  return new AuthError("Could not access private credential storage");
}

/** Run a `node:fs` call, turning anything it throws into the storage access error. */
function access<T>(call: () => T): T {
  try {
    return call();
  } catch {
    throw accessError();
  }
}

/** Whether `error` is a `node:fs` error for a missing path. */
function isMissing(error: unknown): boolean {
  return (error as { code?: unknown } | null)?.code === "ENOENT";
}

/**
 * Check that the open descriptor `fd` is a private directory or regular file; return its status.
 *
 * Throws `AuthError` unless it is the kind asked for (a file with at most one link), owned by
 * this user, and without group or other permission bits.
 */
function checkPrivate(fd: number, directory = false): fs.BigIntStats {
  const info = access(() => fs.fstatSync(fd, { bigint: true }));
  // An atomic replacement can unlink a snapshot already opened by a reader. Its descriptor
  // remains valid; multiple links would expose the secrets.
  if (!(directory ? info.isDirectory() : info.isFile()) || (!directory && info.nlink > 1n)) {
    throw new AuthError(REGULAR);
  }
  if (info.uid !== BigInt(process.getuid!()) || (info.mode & 0o077n) !== 0n) {
    throw new AuthError("Credential storage must be private to its owner");
  }
  return info;
}

/**
 * Return the credentials a stored entry of `method` holds; throw if `value` is not their stored
 * form.
 */
function loadCredentials(method: unknown, value: unknown): Credentials | null {
  if (method === "none") {
    if (value !== null) throw new TypeError("credentials");
    return null;
  }
  if (!isRecord(value)) throw new TypeError("credentials");
  if (method !== "api_key") {
    const optional = Object.hasOwn(value, "residency") ? ["residency"] : [];
    if (!hasKeys(value, [...OAUTH_KEYS, ...optional])) throw new TypeError("credentials");
    const expiresAt = value["expires_at"];
    return new OAuthCredentials({
      accessToken: value["access_token"],
      refreshToken: value["refresh_token"],
      expiresAt: typeof expiresAt === "bigint" ? Number(expiresAt) : expiresAt,
      accountId: value["account_id"],
      residency: value["residency"],
    });
  }
  const hasHeaders = Object.hasOwn(value, "headers");
  if (!hasKeys(value, hasHeaders ? ["api_key", "headers"] : ["api_key"])) throw new TypeError("credentials");
  const headers = hasHeaders ? value["headers"] : {};
  // The writer omits empty headers, so a stored empty mapping is not canonical.
  if (!isRecord(headers) || (hasHeaders && Object.keys(headers).length === 0)) throw new TypeError("headers");
  return new ApiKeyCredentials(value["api_key"], headers);
}

/** Return the provider choice and saved logins `text` holds; throw if it is not a v1 store. */
function parseStore(text: string): [string | null, Providers] {
  const value = parseJson(text, "plain");
  if (!isRecord(value) || !hasKeys(value, STORE_KEYS)) throw new TypeError("store");
  const providerEntries = value["providers"];
  if (value["version"] !== 1n || !isRecord(providerEntries)) throw new TypeError("store");
  const providers: Providers = new Map();
  for (const [provider, entry] of Object.entries(providerEntries)) {
    if (!isRecord(entry)) throw new TypeError("entry");
    const optional = Object.keys(entry).filter((key) => OPTIONAL_ENTRY_KEYS.has(key));
    if (!hasKeys(entry, [...ENTRY_KEYS, ...optional])) throw new TypeError("entry");
    // The writer omits unset ids, so a stored null is not canonical. A null endpoint stays
    // readable, as stores before these ids allowed it.
    if (entry["organization"] === null || entry["project"] === null) throw new TypeError("entry");
    const credentials = loadCredentials(entry["method"], entry["credentials"]);
    const selection = new ProviderSelection({
      provider,
      method: entry["method"],
      model: entry["model"],
      credentialId: entry["credential_id"],
      endpoint: entry["endpoint"],
      organization: entry["organization"],
      project: entry["project"],
      headerNames: credentials instanceof ApiKeyCredentials ? credentials.headers.map(([name]) => name) : [],
    });
    providers.set(provider, [selection, credentials]);
  }
  const fallback = value["default_provider"];
  if (fallback !== null && (typeof fallback !== "string" || !providers.has(fallback))) {
    throw new TypeError("default_provider");
  }
  return [fallback, providers];
}

/**
 * Persist one saved login per provider, with an optional default, privately.
 *
 * A login holds an API key, a subscription, or no credential; an Azure login also holds its
 * endpoint. An API-key login may hold connection settings: extra request headers and, for
 * `openai`, an organization and project id. The store is a JSON file in a directory that, like
 * the file, must be private to its owner and not a symbolic link. Reading a missing store never
 * creates a directory or lock file. The store is POSIX-only.
 *
 * The path is `path`, else `YIAN_AUTH_FILE`, else `<XDG_CONFIG_HOME or ~/.config>/yian/auth.json`,
 * with `~` and a leading `~/` expanded from `HOME` (or the account's home directory) and made
 * absolute. Throws `RangeError` if a timing option is out of range.
 */
export class AuthStore {
  /** The store file's absolute path. */
  readonly path: string;
  readonly #scheduler: Scheduler;
  readonly #signal: AbortSignal | undefined;
  readonly #lockTimeoutMs: number;
  readonly #refreshMarginMs: number;

  constructor(path?: string | null, options: AuthStoreOptions = {}) {
    const {
      env = processEnv,
      scheduler = realScheduler,
      signal,
      lockTimeoutMs = 30_000,
      refreshMarginMs = 60_000,
    } = options;
    const home = () => env("HOME") || userInfo().homedir;
    let target = path ?? (env("YIAN_AUTH_FILE") || null);
    if (target === null) {
      target = join(env("XDG_CONFIG_HOME") || join(home(), ".config"), "yian", "auth.json");
    }
    if (target === "~" || target.startsWith("~/")) target = home() + target.slice(1);
    this.path = resolve(target);
    this.#scheduler = scheduler;
    this.#signal = signal;
    this.#lockTimeoutMs = lockTimeoutMs;
    this.#refreshMarginMs = refreshMarginMs;
    if (
      typeof lockTimeoutMs !== "number" ||
      !(lockTimeoutMs > 0 && lockTimeoutMs <= Number.MAX_VALUE) ||
      typeof refreshMarginMs !== "number" ||
      !(refreshMarginMs >= 0 && refreshMarginMs <= Number.MAX_VALUE)
    ) {
      throw new RangeError("Invalid credential-store timing");
    }
  }

  /**
   * Open and check the store's directory, creating it (0700) first with `create`; return `null`
   * if it is missing and `create` is not set. The caller closes the descriptor.
   */
  #openDirectory(create = false): Directory | null {
    const directory = dirname(this.path);
    if (create) access(() => fs.mkdirSync(directory, { recursive: true, mode: 0o700 }));
    let fd: number;
    try {
      fd = fs.openSync(directory, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW);
    } catch (error) {
      if (!create && isMissing(error)) return null;
      throw accessError();
    }
    try {
      const info = checkPrivate(fd, true);
      return { fd, dev: info.dev, ino: info.ino };
    } catch (error) {
      closeFd(fd);
      throw error;
    }
  }

  /**
   * Throw unless the store's directory path still names the open `directory`.
   *
   * The store file is reached by path, not through the checked directory's descriptor, so a
   * directory swapped in meanwhile would go unchecked.
   */
  #checkSameDirectory(directory: Directory): void {
    const info = access(() => fs.lstatSync(dirname(this.path), { bigint: true }));
    if (info.dev !== directory.dev || info.ino !== directory.ino) throw new AuthError(REGULAR);
  }

  /** Read the default provider and saved logins from the store in `directory`; empty if none. */
  #load(directory: Directory | null): [string | null, Providers] {
    if (directory === null) return [null, new Map()];
    const limit = current().maxAuthStoreBytes;
    let fd: number;
    try {
      fd = fs.openSync(this.path, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
    } catch (error) {
      if (isMissing(error)) return [null, new Map()];
      throw accessError();
    }
    const raw = new Uint8Array(limit + 1);
    let size = 0;
    try {
      this.#checkSameDirectory(directory);
      checkPrivate(fd);
      while (size < raw.length) {
        const n = access(() => fs.readSync(fd, raw, size, raw.length - size, null));
        if (n === 0) break;
        size += n;
      }
    } finally {
      closeFd(fd);
    }
    if (size > limit) throw new AuthError("Credential storage exceeds its size limit");
    try {
      return parseStore(new TextDecoder("utf-8", { fatal: true }).decode(raw.subarray(0, size)));
    } catch {
      throw new AuthError(MALFORMED);
    }
  }

  /** Load the store without the lock, creating nothing. */
  #snapshot(): [string | null, Providers] {
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory();
    try {
      return this.#load(directory);
    } finally {
      if (directory !== null) closeFd(directory.fd);
    }
  }

  /** Read a nonsecret snapshot without creating or changing storage. */
  read(): AuthState {
    const [defaultProvider, providers] = this.#snapshot();
    const selections = Object.freeze(Array.from(providers.values(), ([selection]) => selection));
    return Object.freeze({ defaultProvider, selections });
  }

  /** Find the explicit provider's selection, or the saved default's; `null` if there is none. */
  selection(provider?: string | null): ProviderSelection | null {
    const state = this.read();
    const name = provider ?? state.defaultProvider;
    return state.selections.find((item) => item.provider === name) ?? null;
  }

  /**
   * Return a provider's saved secrets as stored, without refresh or writes.
   *
   * Reads without the lock and returns expired subscription tokens as they are, so callers can
   * scrub every secret the login may send.
   *
   * @returns `[apiKey, ...headerValues]` for an API-key login, `[accessToken, refreshToken,
   *   accountId]` for a subscription, and `[]` for a login without a credential or no login.
   * @throws AuthError The store is malformed or cannot be read.
   */
  savedSecrets(provider: string): readonly string[] {
    const credentials = this.#snapshot()[1].get(provider)?.[1];
    if (credentials instanceof ApiKeyCredentials) {
      return [credentials.apiKey, ...credentials.headers.map(([, value]) => value)];
    }
    if (credentials instanceof OAuthCredentials) {
      return [credentials.accessToken, credentials.refreshToken, credentials.accountId];
    }
    return [];
  }
}

/** Close `fd`, turning a failure into the storage access error. */
function closeFd(fd: number): void {
  access(() => fs.closeSync(fd));
}
