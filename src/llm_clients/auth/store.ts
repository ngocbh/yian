/** Private credential storage: one saved login per provider, with locked atomic writes and refresh. */

import { Buffer } from "node:buffer";
import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { dirname, join, resolve } from "node:path";
import process from "node:process";

import { current, type Env, processEnv } from "../../defaults.ts";
import { compareCodeUnits, shellQuote } from "../../diagnostics.ts";
import {
  FileLockError,
  type FileLock,
  homeDir,
  realScheduler,
  type Scheduler,
  sqliteErrorCode,
  waitForLock,
} from "../../platform.ts";
import { YianRangeError, YianTypeError } from "../../typesys/errors.ts";
import { isRecord } from "../../typesys/index.ts";
import { parseJson } from "../../typesys/json.ts";
import { SUPPORTED_LLM_PROVIDERS } from "../config.ts";
import { OPENAI_API_ENDPOINT } from "../endpoints.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type AuthState,
  type Credentials,
  isUsableLogin,
  OAuthCredentials,
  ProviderSelection,
} from "./common.ts";

// Entry keys written only when set, so a login without them is stored as before.
const OPTIONAL_ENTRY_KEYS = new Set(["endpoint", "organization", "project"]);
const ENTRY_KEYS = ["method", "model", "credential_id", "credentials"];
const STORE_KEYS = ["version", "default_provider", "providers"];
// Written only when set, so a store without a default embedding model reads as before.
const EMBEDDING_KEY = "default_embedding";
const OAUTH_KEYS = ["access_token", "refresh_token", "expires_at", "account_id"];

const OVERSIZED = "Credential storage exceeds its size limit";

/** Saved logins by provider, in file order. */
type Providers = Map<string, [ProviderSelection, Credentials | null]>;

/** What a store holds: the default provider, the logins, and the default embedding model. */
type Saved = [defaultProvider: string | null, providers: Providers, defaultEmbedding: string | null];

/**
 * How a refresh flight ended: its credentials, the locked section's error, or `stopped` when the
 * leader's own store signal aborted it.
 */
type Outcome = { credentials: Credentials } | { error: unknown } | { stopped: true };

// Refreshes in flight in this process, keyed by store path, provider and expected credential id,
// so parallel callers share one refresh and a joiner only receives the login it pinned.
const flights = new Map<string, Promise<Outcome>>();

/** Resolve with `outcome`, or reject with `signal.reason` once `signal` aborts first. */
function joined(outcome: Promise<Outcome>, signal: AbortSignal | undefined): Promise<Outcome> {
  if (signal === undefined) return outcome;
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    void outcome.then((value) => {
      signal.removeEventListener("abort", onAbort);
      resolve(value);
    });
  });
}

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

/**
 * An `AuthError` the store's file, folder or lock causes, so logging in again can't fix it but
 * fixing the file can. `damaged` is set when the file's logins can't be read back, so only
 * restoring the file keeps them.
 *
 * Its name stays "AuthError", as a run failure's `cause_type` shows it; a failed model call
 * records it as `StorageError` (`isStorageError`), so a replay rebuilds it. Its message may name
 * the auth file, so a host relaying errors to others checks `isStorageError` first.
 */
export class StorageError extends AuthError {
  readonly damaged: boolean;

  constructor(message: string, damaged = false) {
    super(message);
    this.damaged = damaged;
  }
}

/**
 * Whether `error` is an auth-store fault (`StorageError`): one the store's file, folder or lock
 * causes, whose message may name the auth file, rather than a sign-in failure.
 */
export function isStorageError(error: unknown): error is StorageError {
  return error instanceof StorageError;
}

/**
 * Whether the saved login `previous` sends what `selection` with `credentials` would: the same
 * method, endpoint, organization and project and, for an API key, the same key and header names
 * and values. A subscription is never the same: a new sign-in brings new tokens. The secrets are
 * compared by digest, in constant time.
 */
function sameLogin(
  [saved, savedCredentials]: [ProviderSelection, Credentials | null],
  selection: ProviderSelection,
  credentials: Credentials | null,
): boolean {
  if (
    saved.method !== selection.method ||
    saved.method === "subscription" ||
    saved.endpoint !== selection.endpoint ||
    saved.organization !== selection.organization ||
    saved.project !== selection.project
  ) {
    return false;
  }
  if (!(savedCredentials instanceof ApiKeyCredentials && credentials instanceof ApiKeyCredentials)) {
    return savedCredentials === null && credentials === null;
  }
  return timingSafeEqual(keyDigest(savedCredentials), keyDigest(credentials));
}

/** Return the sha256 digest of an API key and its headers, sorted by name. */
function keyDigest(credentials: ApiKeyCredentials): Buffer {
  const headers = [...credentials.headers].sort(([a], [b]) => compareCodeUnits(a, b));
  return createHash("sha256").update(JSON.stringify([credentials.apiKey, headers])).digest();
}

/** What a store path names: the store file, its folder, or its lock file. */
type Part = "file" | "folder" | "lock";

/** The start of a message about `part` at `path`. */
function named(part: Part, path: string): string {
  if (part === "folder") return `Credential storage folder ${path}`;
  return part === "lock" ? `Credential storage lock ${path}` : `Credential storage at ${path}`;
}

/**
 * Return the error for `part` at `path` being readable by others. `owned` says whether this user
 * owns it: only then can a chmod fix it; `null` when unknown.
 */
function notPrivate(part: Part, path: string, owned: boolean | null): StorageError {
  // A folder such as /tmp belongs to root: changing its mode would not help, and doing it with
  // sudo would break the system.
  if (owned === false) {
    return new StorageError(
      `${named(part, path)} belongs to another user; keep logins in a folder of your own, such as the ` +
        "default ~/.config/yian",
    );
  }
  const fix = owned ? `; fix it with: chmod ${part === "folder" ? 700 : 600} ${shellQuote(path)}` : "";
  return new StorageError(`${named(part, path)} must be private to its owner${fix}`);
}

/** Return the error for `part` at `path` not being a folder, or a regular file with one link. */
function notRegular(part: Part, path: string): StorageError {
  const kind = part === "folder"
    ? "a folder, not a symbolic link or a file"
    : "a regular file with one link, not a symbolic link";
  return new StorageError(`${named(part, path)} must be ${kind}`);
}

/** Return the error for `part` at `path` being replaced while the store used it. */
function changed(part: Part, path: string): StorageError {
  return new StorageError(`${named(part, path)} changed while in use; try again`);
}

/** Return the error for the store file at `path` whose logins can't be read, `problem` saying why. */
function damaged(path: string, problem: string): StorageError {
  return new StorageError(
    `${named("file", path)} ${problem}; restore it, or delete every saved login in it and log in again ` +
      `(yian auth --remove --auth-file ${shellQuote(path)})`,
    true,
  );
}

/** Return the storage access error; it has no cause, so no path or system detail shows. */
function accessError(): StorageError {
  return new StorageError("Could not access private credential storage");
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

/** Whether `error` is the `node:fs` error a no-follow open gives for a symbolic link, or a file opened as a folder. */
function isWrongKind(error: unknown): boolean {
  const code = (error as { code?: unknown } | null)?.code;
  return code === "ELOOP" || code === "ENOTDIR";
}

/** Whether this user owns `path`, without following a link; `null` if it can't be checked. */
function ownedByUser(path: string): boolean | null {
  try {
    return fs.lstatSync(path, { bigint: true }).uid === BigInt(process.getuid!());
  } catch {
    return null;
  }
}

/** Close `fd`, turning a failure into the storage access error. */
function closeFd(fd: number): void {
  access(() => fs.closeSync(fd));
}

/**
 * Check that the open descriptor `fd`, the store `part` at `path`, is a private folder or regular
 * file; return its status.
 *
 * Throws `StorageError` naming `path` unless it is the kind asked for (a file with at most one
 * link), owned by this user, and without group or other permission bits.
 */
function checkPrivate(fd: number, path: string, part: "file" | "folder"): fs.BigIntStats {
  const info = access(() => fs.fstatSync(fd, { bigint: true }));
  const directory = part === "folder";
  // An atomic replacement can unlink a snapshot already opened by a reader. Its descriptor
  // remains valid; multiple links would expose the secrets.
  if (!(directory ? info.isDirectory() : info.isFile()) || (!directory && info.nlink > 1n)) {
    throw notRegular(part, path);
  }
  const owned = info.uid === BigInt(process.getuid!());
  if (!owned || (info.mode & 0o077n) !== 0n) throw notPrivate(part, path, owned);
  return info;
}

/**
 * Return the credentials a stored entry of `method` holds; throw if `value` is not their stored
 * form.
 */
function loadCredentials(method: unknown, value: unknown): Credentials | null {
  if (method === "none") {
    if (value !== null) throw new YianTypeError("credentials");
    return null;
  }
  if (!isRecord(value)) throw new YianTypeError("credentials");
  if (method !== "api_key") {
    const optional = Object.hasOwn(value, "residency") ? ["residency"] : [];
    if (!hasKeys(value, [...OAUTH_KEYS, ...optional])) throw new YianTypeError("credentials");
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
  if (!hasKeys(value, hasHeaders ? ["api_key", "headers"] : ["api_key"])) throw new YianTypeError("credentials");
  const headers = hasHeaders ? value["headers"] : {};
  // The writer omits empty headers, so a stored empty mapping is not canonical.
  if (!isRecord(headers) || (hasHeaders && Object.keys(headers).length === 0)) throw new YianTypeError("headers");
  return new ApiKeyCredentials(value["api_key"], headers);
}

/** Return the stored form of `credentials`, field by field, so a login without headers keeps its bytes. */
function dumpCredentials(credentials: Credentials | null): Record<string, unknown> | null {
  if (credentials === null) return null;
  if (credentials instanceof ApiKeyCredentials) {
    const value: Record<string, unknown> = { api_key: credentials.apiKey };
    if (credentials.headers.length > 0) value["headers"] = Object.fromEntries(credentials.headers);
    return value;
  }
  return {
    access_token: credentials.accessToken,
    refresh_token: credentials.refreshToken,
    expires_at: credentials.expiresAt,
    account_id: credentials.accountId,
    residency: credentials.residency,
  };
}

/**
 * Serialize `value` (objects, strings, numbers and `null`) as JSON with `", "` and `": "`
 * separators, keys in code-unit order and non-ASCII as lowercase `\uXXXX`. Saved login stores use
 * this format, so it doesn't change. A number is spelled as JS spells it (`1e-7`, `9000`).
 */
function spacedJson(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") {
    // Each code unit outside printable ASCII is written as lowercase \uXXXX.
    return JSON.stringify(value).replace(
      /[^\x20-\x7e]/g,
      (unit) => `\\u${unit.charCodeAt(0).toString(16).padStart(4, "0")}`,
    );
  }
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => compareCodeUnits(a, b));
  return `{${entries.map(([key, item]) => `${spacedJson(key)}: ${spacedJson(item)}`).join(", ")}}`;
}

/** Return the provider choice, saved logins and embedding model `text` holds; throw if it is not a v1 store. */
function parseStore(text: string): Saved {
  const value = parseJson(text, "plain");
  if (!isRecord(value)) throw new YianTypeError("store");
  const embedding = value[EMBEDDING_KEY];
  if (!hasKeys(value, embedding === undefined ? STORE_KEYS : [...STORE_KEYS, EMBEDDING_KEY])) {
    throw new YianTypeError("store");
  }
  if (embedding !== undefined && typeof embedding !== "string") throw new YianTypeError(EMBEDDING_KEY);
  const providerEntries = value["providers"];
  if (value["version"] !== 1n || !isRecord(providerEntries)) throw new YianTypeError("store");
  const providers: Providers = new Map();
  for (const [provider, entry] of Object.entries(providerEntries)) {
    if (!isRecord(entry)) throw new YianTypeError("entry");
    const optional = Object.keys(entry).filter((key) => OPTIONAL_ENTRY_KEYS.has(key));
    if (!hasKeys(entry, [...ENTRY_KEYS, ...optional])) throw new YianTypeError("entry");
    // The writer omits unset ids, so a stored null is not canonical. A null endpoint stays
    // readable, as stores before these ids allowed it.
    if (entry["organization"] === null || entry["project"] === null) throw new YianTypeError("entry");
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
    throw new YianTypeError("default_provider");
  }
  return [fallback, providers, embedding ?? null];
}

/**
 * Persist one saved login per provider, with an optional default, and an optional default
 * embedding model, privately.
 *
 * A login holds an API key, a subscription, or no credential; an Azure login also holds its
 * endpoint. An API-key login may hold connection settings: extra request headers and, for
 * `openai`, an organization and project id. The store is a JSON file in a directory that, like
 * the file, must be private to its owner and not a symbolic link. Reading a missing store never
 * creates a directory or lock file. Selection and credentials are committed together by `save`;
 * `setDefaultEmbedding` also creates a store, while `remove`, `setDefault` and `delete` change a
 * saved store and never create one. Writes and
 * refreshes hold the `<path>.lock` file lock; a refresh reloads under it and cannot replace a newer
 * connection. The store is POSIX-only.
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
    const home = () => homeDir(env);
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
      throw new YianRangeError("Invalid credential-store timing");
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
      if (isWrongKind(error)) throw notRegular("folder", directory);
      throw accessError();
    }
    try {
      const info = checkPrivate(fd, directory, "folder");
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
    if (info.dev !== directory.dev || info.ino !== directory.ino) throw changed("folder", dirname(this.path));
  }

  /** Read what the store in `directory` holds; empty if none. */
  #load(directory: Directory | null): Saved {
    if (directory === null) return [null, new Map(), null];
    const limit = current().maxAuthStoreBytes;
    let fd: number;
    try {
      fd = fs.openSync(this.path, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
    } catch (error) {
      if (isWrongKind(error)) throw notRegular("file", this.path);
      if (!isMissing(error)) throw accessError();
      // A missing file in a swapped-in directory says nothing about the checked one.
      this.#checkSameDirectory(directory);
      return [null, new Map(), null];
    }
    let raw: Uint8Array;
    let size = 0;
    try {
      this.#checkSameDirectory(directory);
      const info = checkPrivate(fd, this.path, "file");
      // Sized to the file and one byte more, which shows the file grew: then the read goes on, up
      // to one byte past the limit.
      raw = new Uint8Array(Math.min(Number(info.size), limit) + 1);
      for (;;) {
        if (size === raw.length) {
          if (raw.length > limit) break;
          const grown = new Uint8Array(Math.min(raw.length * 2, limit + 1));
          grown.set(raw);
          raw = grown;
        }
        const n = access(() => fs.readSync(fd, raw, size, raw.length - size, null));
        if (n === 0) break;
        size += n;
      }
    } finally {
      closeFd(fd);
    }
    if (size > limit) throw damaged(this.path, "exceeds its size limit");
    try {
      return parseStore(new TextDecoder("utf-8", { fatal: true }).decode(raw.subarray(0, size)));
    } catch {
      throw damaged(this.path, "is malformed");
    }
  }

  /** Load the store without the lock, creating nothing. */
  #snapshot(): Saved {
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory();
    try {
      return this.#load(directory);
    } finally {
      if (directory !== null) closeFd(directory.fd);
    }
  }

  /**
   * Run `body` holding the store's lock, waiting for it up to the lock timeout.
   *
   * The lock file is only ever touched through `waitForLock`: closing any other descriptor on it
   * would drop this process's lock.
   */
  async #locked<T>(body: () => T | Promise<T>): Promise<T> {
    const path = `${this.path}.lock`;
    let lock: FileLock;
    try {
      lock = await waitForLock(path, {
        private: true,
        timeoutMs: this.#lockTimeoutMs,
        signal: this.#signal,
        scheduler: this.#scheduler,
      });
    } catch (error) {
      if (this.#signal?.aborted && error === this.#signal.reason) throw error;
      if (!(error instanceof FileLockError)) throw accessError();
      if (error.reason === "busy") throw new StorageError("Timed out waiting for credential storage");
      if (error.reason === "not-private") throw notPrivate("lock", path, ownedByUser(path));
      if (error.reason === "changed") throw changed("lock", path);
      // A lock file SQLite can't read is a failure of the lock itself, not of the path's kind.
      throw sqliteErrorCode(error.cause) === undefined ? notRegular("lock", path) : accessError();
    }
    try {
      return await body();
    } finally {
      lock.release();
    }
  }

  /**
   * Atomically replace the store in `directory` with `saved`.
   *
   * Checks that the directory path still names `directory`, writes a private temporary file
   * beside the store, then renames it over the store and checks that the store path and its
   * directory still name what was written and checked.
   */
  #write(directory: Directory, [defaultProvider, providers, defaultEmbedding]: Saved): void {
    const value: Record<string, unknown> = {
      version: 1,
      default_provider: defaultProvider,
      providers: Object.fromEntries(
        Array.from(providers, ([provider, [selection, credentials]]) => {
          const entry: Record<string, unknown> = {
            method: selection.method,
            model: selection.model,
            credential_id: selection.credentialId,
            credentials: dumpCredentials(credentials),
          };
          for (const key of ["endpoint", "organization", "project"] as const) {
            if (selection[key] !== null) entry[key] = selection[key];
          }
          return [provider, entry];
        }),
      ),
    };
    if (defaultEmbedding !== null) value[EMBEDDING_KEY] = defaultEmbedding;
    const raw = new TextEncoder().encode(`${spacedJson(value)}\n`);
    if (raw.length > current().maxAuthStoreBytes) throw new AuthError(OVERSIZED);
    // Checked before the temporary file exists, so a swapped-in directory never receives it.
    this.#checkSameDirectory(directory);
    const temporary = join(dirname(this.path), `.auth-${randomBytes(16).toString("hex")}`);
    const { O_CREAT, O_EXCL, O_NOFOLLOW, O_WRONLY } = fs.constants;
    const fd = access(() => fs.openSync(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600));
    try {
      let written: fs.BigIntStats;
      try {
        written = access(() => fs.fstatSync(fd, { bigint: true }));
        if (written.dev !== directory.dev) throw changed("folder", dirname(this.path));
        let offset = 0;
        while (offset < raw.length) offset += access(() => fs.writeSync(fd, raw, offset));
        access(() => fs.fsyncSync(fd));
      } finally {
        closeFd(fd);
      }
      this.#signal?.throwIfAborted();
      access(() => fs.renameSync(temporary, this.path));
      const target = access(() => fs.lstatSync(this.path, { bigint: true }));
      if (target.dev !== written.dev || target.ino !== written.ino) throw changed("file", this.path);
      this.#checkSameDirectory(directory);
      access(() => fs.fsyncSync(directory.fd));
    } finally {
      try {
        fs.unlinkSync(temporary);
      } catch (error) {
        if (!isMissing(error)) throw accessError();
      }
    }
  }

  /** Read a nonsecret snapshot without creating or changing storage. */
  read(): AuthState {
    const [defaultProvider, providers, defaultEmbedding] = this.#snapshot();
    const selections = Object.freeze(Array.from(providers.values(), ([selection]) => selection));
    return Object.freeze({ defaultProvider, selections, defaultEmbedding });
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

  /**
   * Atomically reconnect a provider and optionally make it the default.
   *
   * The new login replaces the old one with its settings: settings not given here are not kept.
   * `credentials` is the validated credential (an API key's headers are saved with it), or `null`
   * for a provider that needs none; `endpoint` is the provider's base URL, such as an Azure
   * resource, or the endpoint an `openai` API key is for (default OpenAI's API); `organization`
   * and `project` are OpenAI ids, for an `openai` API key only; `makeDefault` (default `true`)
   * makes the provider the default.
   *
   * A login saved again unchanged keeps its credential id, so runs pinned to it go on: same method,
   * key, header names and values, endpoint, organization and project, whatever the model. Any
   * other save, a subscription's included, gets a new id.
   *
   * @returns The saved selection.
   * @throws AuthError The values are invalid or storage cannot be written.
   */
  async save(
    provider: string,
    credentials: Credentials | null,
    model: string,
    options: {
      endpoint?: string | null;
      organization?: string | null;
      project?: string | null;
      makeDefault?: boolean;
    } = {},
  ): Promise<ProviderSelection> {
    const { organization = null, project = null, makeDefault = true } = options;
    const openaiKey = provider === "openai" && credentials instanceof ApiKeyCredentials;
    const endpoint = options.endpoint === undefined ? (openaiKey ? OPENAI_API_ENDPOINT : null) : options.endpoint;
    // Exactly these classes, not subclasses: a subclass could change what is stored.
    if (
      credentials !== null &&
      !(
        typeof credentials === "object" &&
        (Object.getPrototypeOf(credentials) === ApiKeyCredentials.prototype ||
          Object.getPrototypeOf(credentials) === OAuthCredentials.prototype)
      )
    ) {
      throw new AuthError("Unsupported credential type");
    }
    const fields = {
      provider,
      method: credentials === null ? "none" : credentials.method,
      model,
      endpoint,
      organization,
      project,
      headerNames: credentials instanceof ApiKeyCredentials ? credentials.headers.map(([name]) => name) : [],
    };
    let selection = new ProviderSelection({ ...fields, credentialId: randomBytes(16).toString("hex") });
    // The snapshot refuses such a login, so saving it would make every run throw. A provider this
    // version doesn't run is never read, so its login isn't checked.
    if ((SUPPORTED_LLM_PROVIDERS as readonly string[]).includes(provider) && !isUsableLogin(selection)) {
      throw new AuthError(`A ${provider} login can't be saved with this credential or endpoint`);
    }
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory(true)!;
    try {
      await this.#locked(() => {
        const [defaultProvider, providers, defaultEmbedding] = this.#load(directory);
        const previous = providers.get(provider);
        if (previous !== undefined && sameLogin(previous, selection, credentials)) {
          selection = new ProviderSelection({ ...fields, credentialId: previous[0].credentialId });
        }
        providers.set(provider, [selection, credentials]);
        this.#write(directory, [makeDefault ? provider : defaultProvider, providers, defaultEmbedding]);
      });
    } finally {
      closeFd(directory.fd);
    }
    return selection;
  }

  /** Remove `provider`'s saved login, or make it the default; throw if it has none. */
  async #update(provider: string, remove: boolean): Promise<void> {
    const missing = () => new AuthError(`There is no saved ${provider} login; run 'yian auth' to list them`);
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory();
    if (directory === null) throw missing();
    try {
      await this.#locked(() => {
        let [defaultProvider, providers, defaultEmbedding] = this.#load(directory);
        if (!providers.has(provider)) throw missing();
        if (remove) {
          providers.delete(provider);
          if (defaultProvider === provider) defaultProvider = null;
        } else {
          defaultProvider = provider;
        }
        this.#write(directory, [defaultProvider, providers, defaultEmbedding]);
      });
    } finally {
      closeFd(directory.fd);
    }
  }

  /**
   * Delete a provider's saved login, clearing the default if it names it.
   *
   * @throws AuthError The provider has no saved login or storage fails.
   */
  remove(provider: string): Promise<void> {
    return this.#update(provider, true);
  }

  /**
   * Make a saved provider the default.
   *
   * @throws AuthError The provider has no saved login or storage fails.
   */
  setDefault(provider: string): Promise<void> {
    return this.#update(provider, false);
  }

  /**
   * Save `embedding`, a `provider/model` the caller checked, as the model a new VectorDB memory is
   * made with, creating the store if there is none.
   *
   * @throws AuthError Storage fails.
   */
  async setDefaultEmbedding(embedding: string): Promise<void> {
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory(true)!;
    try {
      await this.#locked(() => {
        const [defaultProvider, providers] = this.#load(directory);
        this.#write(directory, [defaultProvider, providers, embedding]);
      });
    } finally {
      closeFd(directory.fd);
    }
  }

  /**
   * Delete the whole saved store under its lock, keeping the lock file.
   *
   * Creates nothing when neither the store nor its lock file exists. A symbolic link at the
   * store's path is removed, never followed.
   *
   * @throws AuthError The directory is not private, the lock wait times out, or storage fails.
   */
  async delete(): Promise<void> {
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory();
    if (directory === null) return;
    try {
      const exists = (path: string) => {
        try {
          fs.lstatSync(path);
          return true;
        } catch (error) {
          if (isMissing(error)) return false;
          throw accessError();
        }
      };
      if (!exists(this.path) && !exists(`${this.path}.lock`)) {
        // Missing files in a swapped-in directory say nothing about the checked one.
        this.#checkSameDirectory(directory);
        return;
      }
      // Unlinking the lock file would let a waiter and a newcomer lock different files, so only
      // the store goes.
      await this.#locked(() => {
        // The unlink goes by path, not through the checked directory's descriptor, so check the
        // path still names that directory, as `#write` does.
        this.#checkSameDirectory(directory);
        try {
          fs.unlinkSync(this.path);
        } catch (error) {
          if (isMissing(error)) return;
          throw accessError();
        }
        access(() => fs.fsyncSync(directory.fd));
      });
    } finally {
      closeFd(directory.fd);
    }
  }

  /** Start the refresh flight `key` running `locked`, registered until it settles. */
  #lead(key: string, locked: () => Promise<Credentials>): Promise<Outcome> {
    const flight = locked()
      .then(
        (credentials): Outcome => ({ credentials }),
        (error: unknown): Outcome =>
          this.#signal?.aborted && error === this.#signal.reason ? { stopped: true } : { error },
      )
      .finally(() => {
        if (flights.get(key) === flight) flights.delete(key);
      });
    flights.set(key, flight);
    return flight;
  }

  /**
   * Resolve a pinned connection, refreshing expiring subscription tokens under the lock.
   *
   * An API key, or a subscription token valid beyond the refresh margin, returns as saved, read
   * without the lock. Otherwise this takes the lock, reloads, and, unless another caller has
   * refreshed meanwhile, saves and returns `refresh(current)`, which must keep the account id and
   * be fresh. `refresh`'s own errors propagate unchanged.
   *
   * Parallel calls in this process for the same store path, provider and credential id share one
   * refresh flight: the first runs it, the others wait for its credentials or rethrow its error
   * (a failed refresh is not retried). A waiting caller has no timeout of its own: it waits until
   * the flight settles or its store signal aborts. If the leader's signal aborts, the leader
   * rejects with its reason and the waiting callers start over.
   *
   * @throws AuthError The saved login is no longer `expectedCredentialId`, has no credential, has
   *   expired without a `refresh`, the refresh is refused, or storage fails.
   */
  async credentials(
    provider: string,
    expectedCredentialId: string,
    options: { refresh?: (current: OAuthCredentials) => OAuthCredentials | Promise<OAuthCredentials> } = {},
  ): Promise<Credentials> {
    const { refresh } = options;
    const pinned = (providers: Providers) => {
      const pair = providers.get(provider);
      if (pair === undefined || pair[0].credentialId !== expectedCredentialId) {
        throw new AuthError("The saved connection changed");
      }
      if (pair[1] === null) throw new AuthError(`${provider} has no saved credential`);
      return [pair[0], pair[1]] as const;
    };
    const fresh = (credentials: Credentials) =>
      credentials instanceof ApiKeyCredentials ||
      credentials.expiresAt * 1000 > this.#scheduler.now() + this.#refreshMarginMs;
    this.#signal?.throwIfAborted();
    const directory = this.#openDirectory();
    try {
      const [, saved] = pinned(this.#load(directory)[1]);
      if (fresh(saved)) return saved;
      // A saved login means the directory exists.
      const locked = () => this.#locked(async () => {
        const [defaultProvider, providers, defaultEmbedding] = this.#load(directory!);
        const [selection, credentials] = pinned(providers);
        if (fresh(credentials)) return credentials;
        if (refresh === undefined) throw new AuthError("Subscription credentials expired; sign in again");
        this.#signal?.throwIfAborted();
        const expired = credentials as OAuthCredentials;
        const updated: unknown = await refresh(expired);
        if (
          typeof updated !== "object" ||
          updated === null ||
          Object.getPrototypeOf(updated) !== OAuthCredentials.prototype ||
          (updated as OAuthCredentials).accountId !== expired.accountId
        ) {
          throw new AuthError("Subscription refresh changed account identity");
        }
        const checked = updated as OAuthCredentials;
        if (!fresh(checked)) throw new AuthError("Subscription refresh returned expired credentials");
        providers.set(provider, [selection, checked]);
        this.#write(directory!, [defaultProvider, providers, defaultEmbedding]);
        return checked;
      });
      // Without a refresh the locked section only returns a fresh token or fails: nothing to share.
      if (refresh === undefined) return await locked();
      const key = [this.path, provider, expectedCredentialId].join("\0");
      for (;;) {
        const existing = flights.get(key);
        const outcome = await (existing === undefined ? this.#lead(key, locked) : joined(existing, this.#signal));
        if ("credentials" in outcome) return outcome.credentials;
        // Never refresh again after a failure: a refresh that timed out may have rotated the
        // token, and reusing the old one can revoke the login.
        if ("error" in outcome) throw outcome.error;
        if (existing === undefined) throw this.#signal!.reason;
        // Only the leader's own stop ended the flight, so the token is still due: start over, and
        // the first joiner back leads the next flight. A stop during the leader's refresh request
        // can't tell whether the server already rotated the token; the next flight sends the saved
        // one.
      }
    } finally {
      if (directory !== null) closeFd(directory.fd);
    }
  }
}
