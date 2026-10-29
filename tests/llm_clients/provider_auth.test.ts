/**
 * `auth/index.ts`'s exports, the credential values, the store, `isAuthFailure`, `request`,
 * `AuthError`'s status, and `OpenAIAuth`'s sign-in flows, refresh and model catalogs; and the
 * scenarios of `auth_both_runtimes.ts`, on Bun and, with `YIAN_TEST_NODE`, built on Node.
 */

import { type ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import net, { type AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { inspect } from "node:util";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import { ENTRIES } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type Credentials,
  HttpResponse,
  isAuthFailure,
  OAuthCredentials,
  ProviderSelection,
} from "../../src/llm_clients/auth/common.ts";
import { request } from "../../src/llm_clients/auth/http.ts";
import { type AuthRequest, OpenAIAuth } from "../../src/llm_clients/auth/openai.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { lockFile, realScheduler, type Scheduler, USER_AGENT, virtualScheduler } from "../../src/platform.ts";
import { parseJson } from "../../src/typesys/json.ts";
import { SCENARIOS } from "./auth_both_runtimes.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

const PLATFORM_URL = pathToFileURL(join(import.meta.dirname, "..", "..", "src", "platform.ts")).href;

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** Return the error `promise` rejects with, failing the test if it resolves. */
async function rejected(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

test("auth/index.ts exports the auth package names", async () => {
  // The types (AuthState, Credentials) have no runtime value.
  const names = Object.keys(await import("../../src/llm_clients/auth/index.ts")).sort();
  expect(names).toEqual([
    "ApiKeyCredentials",
    "AuthError",
    "AuthStore",
    "HttpResponse",
    "OAuthCredentials",
    "OpenAIAuth",
    "ProviderSelection",
    "isAuthFailure",
  ]);
});

describe("ApiKeyCredentials", () => {
  const refusedHeaders: [string, unknown][] = [
    ["Authorization", { Authorization: "do-not-echo" }],
    ["x-api-key", { "x-api-key": "do-not-echo" }],
    ["Content-Type", { "Content-Type": "do-not-echo" }],
    ["bad name", { "bad name": "do-not-echo" }],
    ["CRLF value", { X: "do-not-echo\r\nY: z" }],
    ["leading space", { X: " do-not-echo" }],
    ["empty value", { X: "" }],
    ["number value", { X: 1 }],
    ["case repeat", { "X-G": "do-not-echo", "x-g": "other" }],
    [
      "exact repeat",
      [
        ["X-G", "do-not-echo"],
        ["X-G", "other"],
      ],
    ],
    ["33 headers", Object.fromEntries(Array.from({ length: 33 }, (_, n) => [`X-${n}`, "v"]))],
    ["a string", "X-G"],
    ["number name", [[1, "do-not-echo"]]],
    ["symbol name", [[Symbol("X"), "do-not-echo"]]],
    ["three-item pair", [["X", "do-not-echo", "extra"]]],
    ["null", null],
  ];
  for (const [label, headers] of refusedHeaders) {
    test(`refuse headers that cannot be sent: ${label}`, () => {
      const error = thrown(() => new ApiKeyCredentials("k", headers));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).not.toContain("do-not-echo");
    });
  }

  test("refusal messages", () => {
    expect(thrown(() => new ApiKeyCredentials("k", "X-G")).message).toBe(
      "Invalid connection headers: give a mapping of name to value",
    );
    expect(
      thrown(
        () =>
          new ApiKeyCredentials("k", [
            ["X-G", "a"],
            ["X-G", "b"],
          ]),
      ).message,
    ).toBe("Invalid connection headers: a header name is given more than once");
    expect(thrown(() => new ApiKeyCredentials("k", { Authorization: "a" })).message).toBe(
      "Invalid connection headers: header Authorization is refused: yian sends the credential itself",
    );
    expect(thrown(() => new ApiKeyCredentials("k", [["X-A", undefined]])).message).toBe(
      "Invalid connection headers: header X-A value must be a string",
    );
    expect(thrown(() => new ApiKeyCredentials("k", new Map([["X-A", undefined]]))).message).toBe(
      "Invalid connection headers: header X-A value must be a string",
    );
  });

  test("the key must be visible text", () => {
    for (const key of ["", "has space", "ké", 1, null]) {
      const error = thrown(() => new ApiKeyCredentials(key));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Invalid API key credential");
    }
  });

  test("hold sorted immutable headers", () => {
    const credentials = new ApiKeyCredentials("k", { "X-B": "2", "X-A": "1" });
    expect(credentials.headers).toEqual([
      ["X-A", "1"],
      ["X-B", "2"],
    ]);
    const fromPairs = new ApiKeyCredentials("k", [
      ["X-B", "2"],
      ["X-A", "1"],
    ]);
    expect(fromPairs.apiKey).toBe(credentials.apiKey);
    expect(fromPairs.headers).toEqual(credentials.headers);
    expect(new ApiKeyCredentials("k").headers).toEqual([]);
    expect(new ApiKeyCredentials("k", { "OpenAI-Project": "p" }).headers).toEqual([["OpenAI-Project", "p"]]);
    expect(Object.isFrozen(credentials)).toBe(true);
    expect(Object.isFrozen(credentials.headers)).toBe(true);
    expect(credentials.headers.every((pair) => Object.isFrozen(pair))).toBe(true);
    expect(credentials.method).toBe("api_key");
  });

  test("a Map of headers is accepted and undefined record entries are skipped", () => {
    const credentials = new ApiKeyCredentials(
      "k",
      new Map([
        ["X-B", "2"],
        ["X-A", "1"],
      ]),
    );
    expect(credentials.headers).toEqual([
      ["X-A", "1"],
      ["X-B", "2"],
    ]);
    expect(new ApiKeyCredentials("k", { "X-A": "1", "X-B": undefined }).headers).toEqual([["X-A", "1"]]);
  });

  test("the key and header values never show", () => {
    const credentials = new ApiKeyCredentials("sk-secret-key", { "X-Gateway": "gw-secret" });
    for (const text of [JSON.stringify(credentials), String(credentials), inspect(credentials)]) {
      expect(text).not.toContain("sk-secret-key");
      expect(text).not.toContain("gw-secret");
    }
    expect(credentials.apiKey).toBe("sk-secret-key");
  });
});

describe("OAuthCredentials", () => {
  const fields = { accessToken: "access", refreshToken: "refresh", expiresAt: 1000, accountId: "acct" };

  test("keeps tokens and routing metadata", () => {
    const credentials = new OAuthCredentials({ ...fields, residency: "us" });
    expect(credentials.method).toBe("subscription");
    expect(credentials.accessToken).toBe("access");
    expect(credentials.refreshToken).toBe("refresh");
    expect(credentials.expiresAt).toBe(1000);
    expect(credentials.accountId).toBe("acct");
    expect(credentials.residency).toBe("us");
    expect(new OAuthCredentials(fields).residency).toBeNull();
    expect(new OAuthCredentials({ ...fields, expiresAt: Number.MAX_VALUE }).expiresAt).toBe(Number.MAX_VALUE);
    expect(Object.isFrozen(credentials)).toBe(true);
  });

  for (const [label, expiry] of [
    ["10n ** 1000n", 10n ** 1000n],
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
    ["true", true],
    ["0", 0],
  ] as const) {
    test(`invalid expiry is rejected without unhandled conversion errors: ${label}`, () => {
      const error = thrown(() => new OAuthCredentials({ ...fields, expiresAt: expiry }));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Invalid subscription credentials");
    });
  }

  test("tokens, account id and residency must be visible text", () => {
    for (const bad of [
      { accessToken: "" },
      { refreshToken: "has space" },
      { accountId: 1 },
      { residency: "" },
      { residency: "eú" },
    ]) {
      expect(thrown(() => new OAuthCredentials({ ...fields, ...bad })).message).toBe(
        "Invalid subscription credentials",
      );
    }
  });

  test("tokens never show", () => {
    const credentials = new OAuthCredentials({ ...fields, accessToken: "at-secret", refreshToken: "rt-secret" });
    for (const text of [JSON.stringify(credentials), String(credentials), inspect(credentials)]) {
      expect(text).not.toContain("at-secret");
      expect(text).not.toContain("rt-secret");
    }
  });
});

describe("ProviderSelection", () => {
  test("checks connection settings", () => {
    const selection = new ProviderSelection({
      provider: "openai",
      method: "api_key",
      model: "m",
      credentialId: "c".repeat(32),
      endpoint: null,
      organization: "org-1",
      project: "proj-1",
      headerNames: ["X-B", "X-A"],
    });
    expect(selection.headerNames).toEqual(["X-A", "X-B"]);
    expect(Object.isFrozen(selection)).toBe(true);
    expect(Object.isFrozen(selection.headerNames)).toBe(true);
    const plain = new ProviderSelection({ provider: "p", method: "api_key", model: "m", credentialId: "c".repeat(32) });
    expect(plain.headerNames).toEqual([]);
    expect(plain.endpoint).toBeNull();
    expect(plain.organization).toBeNull();
    expect(plain.project).toBeNull();
  });

  const refusals: [string, Record<string, unknown>, string][] = [
    ["empty provider", { provider: "" }, "Invalid provider selection"],
    ["unknown method", { method: "other" }, "Invalid provider selection"],
    ["endpoint with space", { endpoint: "has space" }, "Invalid provider selection"],
    ["headerNames a string", { headerNames: "X-A" }, "Invalid provider selection"],
    [
      "subscription with organization",
      { method: "subscription", organization: "org-1" },
      "organization and project are only for an openai API-key login",
    ],
    [
      "anthropic with project",
      { provider: "anthropic", project: "proj-1" },
      "organization and project are only for an openai API-key login",
    ],
    [
      "subscription with headers",
      { method: "subscription", headerNames: ["X-A"] },
      "Connection headers are only for an API-key login",
    ],
    ["none with headers", { method: "none", headerNames: ["X-A"] }, "Connection headers are only for an API-key login"],
    [
      "ollama with headers",
      { provider: "ollama", headerNames: ["X-A"] },
      "Connection headers are not sent to provider ollama",
    ],
    [
      "organization with space",
      { organization: "has space" },
      "Invalid provider selection: organization must be non-empty visible ASCII, without spaces",
    ],
    [
      "Authorization header",
      { headerNames: ["Authorization"] },
      "Invalid provider selection: header Authorization is refused: yian sends the credential itself",
    ],
    [
      "OpenAI-Project header",
      { headerNames: ["OpenAI-Project"] },
      "Invalid provider selection: header OpenAI-Project is refused: use llm_config.organization / project",
    ],
    [
      "case-repeated header",
      { headerNames: ["X-A", "x-a"] },
      "Invalid provider selection: a header name is given more than once (names ignore case)",
    ],
    [
      "33 headers",
      { headerNames: Array.from({ length: 33 }, (_, n) => `X-${n}`) },
      "Invalid provider selection: headers must have at most 32 headers",
    ],
  ];
  for (const [label, fields, message] of refusals) {
    test(`refuses ${label}`, () => {
      const values = { provider: "openai", method: "api_key", model: "m", credentialId: "c".repeat(32), ...fields };
      const error = thrown(() => new ProviderSelection(values));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(message);
    });
  }
});

/** A provider error carrying an HTTP `status`, like the openai SDK's `APIStatusError`. */
class StatusError extends Error {
  readonly status: unknown;

  constructor(message: string, status: unknown) {
    super(message);
    this.status = status;
  }
}

/** A provider error carrying an HTTP `code`, like google-genai's `ClientError`. */
class CodeError extends Error {
  readonly code: unknown;

  constructor(message: string, code: unknown) {
    super(message);
    this.code = code;
  }
}

/** A 400 whose text can't be read. */
class UnreadableError extends StatusError {
  override toString(): string {
    throw new Error("no text");
  }
}

/** An error whose `status` getter throws. */
class RaisingStatus extends Error {
  get status(): number {
    throw new Error("no status");
  }
}

/** Return an error whose `cause` getter throws. */
function raisingCause(): Error {
  return Object.defineProperty(new Error("wrapper"), "cause", {
    get() {
      throw new Error("no cause");
    },
  });
}

/** Return two errors that are each other's `cause`. */
function cycle(): Error {
  const first = new Error("first");
  const second = new Error("second", { cause: first });
  first.cause = second;
  return first;
}

/** Return a `Proxy` of an `AuthError` whose `getPrototypeOf` trap throws, so `instanceof` throws. */
function prototypeTrap(): object {
  return new Proxy(new AuthError("no credentials"), {
    getPrototypeOf() {
      throw new Error("no prototype");
    },
  });
}

/** Return a 401 behind `depth` plain wrapping errors. */
function wrapped401(depth: number): Error {
  let error: Error = new StatusError("unauthorized", 401);
  for (let n = 0; n < depth; n += 1) error = new Error(`wrapper ${n}`, { cause: error });
  return error;
}

describe("isAuthFailure", () => {
  const rows: [string, unknown, boolean][] = [
    ["auth-error", new AuthError("Subscription credentials expired"), true],
    ["401", new StatusError("unauthorized", 401), true],
    ["403", new StatusError("forbidden", 403), true],
    ["400", new StatusError("bad request", 400), false],
    ["404", new StatusError("not found", 404), false],
    ["429", new StatusError("rate limited", 429), false],
    ["500", new StatusError("server error", 500), false],
    ["wrapped-401", new Error("wrapped", { cause: new StatusError("unauthorized", 401) }), true],
    ["code-401", new CodeError("unauthenticated", 401), true],
    ["wrapped-auth-error", new Error("model build", { cause: new AuthError("no credentials") }), true],
    ["400-api-key-invalid", new StatusError("API key not valid. reason: API_KEY_INVALID", 400), true],
    ["code-400-api-key-invalid", new CodeError("API key not valid. reason: API_KEY_INVALID", 400), true],
    ["400-other", new StatusError("invalid argument", 400), false],
    ["bool-status", new StatusError("unauthorized", true), false],
    ["value-error", new RangeError("bad value"), false],
    ["unreadable-text", new UnreadableError("unused", 400), false],
    ["raising-status", new RaisingStatus("unreadable status"), false],
    ["cause-cycle", cycle(), false],
    ["16th link", wrapped401(15), true],
    ["17th link", wrapped401(16), false],
    ["raising-cause", raisingCause(), false],
    ["throwing-instanceof", prototypeTrap(), false],
    ["provider-error-auth", new ProviderError("auth", "API key not valid", { status: 400 }), true],
    ["provider-error-bad-request", new ProviderError("bad_request", "invalid argument", { status: 400 }), false],
    // A content-filter 403 (OpenRouter's content-policy reply) is a refusal, not an auth failure.
    ["provider-error-refused-403", new ProviderError("refused", "flagged", { status: 403 }), false],
    ["provider-error-permission-403", new ProviderError("permission", "forbidden", { status: 403 }), true],
    ["not an object", "401", false],
    ["null", null, false],
  ];
  for (const [label, error, expected] of rows) {
    test(label, () => {
      expect(isAuthFailure(error)).toBe(expected);
    });
  }
});

describe("the test preload", () => {
  const seen: { authFile: string; home: string }[] = [];

  /** Check this test's store and home and the cleared variables; record the store and home. */
  function checkIsolated(): void {
    const authFile = process.env.YIAN_AUTH_FILE;
    const home = process.env.HOME;
    if (authFile === undefined || home === undefined) throw new Error("YIAN_AUTH_FILE and HOME must be set");
    expect(dirname(authFile).startsWith(tmpdir())).toBe(true);
    expect(fs.existsSync(authFile)).toBe(false);
    expect(new AuthStore().path).toBe(authFile);
    expect(new AuthStore(null, { env: NO_ENV }).path).toBe(authFile);
    expect(home.startsWith(tmpdir())).toBe(true);
    expect(fs.readdirSync(home)).toEqual([]);
    expect(process.env.XDG_CONFIG_HOME).toBeUndefined();
    const cleared = [...ENTRIES.map((each) => each.variable), "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL"];
    expect(cleared.filter((name) => process.env[name] !== undefined)).toEqual([]);
    expect(process.env.YIAN_PRELOAD_LEAK).toBeUndefined();
    for (const before of seen) {
      expect(dirname(authFile)).not.toBe(dirname(before.authFile));
      expect(home).not.toBe(before.home);
    }
    seen.push({ authFile, home });
  }

  test("gives a test an empty store at YIAN_AUTH_FILE, the default path also through NO_ENV, and an empty HOME", () => {
    checkIsolated();
    // The next test sees neither this variable nor this store.
    process.env.YIAN_PRELOAD_LEAK = "1";
  });

  test("gives the next test a fresh store and home and drops what the last one set", () => {
    checkIsolated();
  });
});

describe("AuthStore", () => {
  let tmp: string;
  let file: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(join(tmpdir(), "yian-auth-"));
    file = join(tmp, "auth", "auth.json");
  });

  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  /** Write `value` as the store at `path`: its directory 0700, the file 0600. */
  function writeStore(path: string, value: unknown): void {
    writeRaw(path, JSON.stringify(value));
  }

  /** Write `contents` as the store at `path`: its directory 0700, the file 0600. */
  function writeRaw(path: string, contents: string | Uint8Array): void {
    const directory = dirname(path);
    fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
    fs.chmodSync(directory, 0o700);
    fs.writeFileSync(path, contents);
    fs.chmodSync(path, 0o600);
  }

  /** Return a one-provider store's text, the entry overridden by `entry`. */
  function entryStore(provider = "p", entry: Record<string, unknown> = {}): string {
    const value = {
      method: "api_key",
      model: "m",
      credential_id: "c".repeat(32),
      credentials: { api_key: "do-not-echo" },
      ...entry,
    };
    return JSON.stringify({ version: 1, default_provider: null, providers: { [provider]: value } });
  }

  /** A saved openai API-key login. */
  const savedStore = {
    version: 1,
    default_provider: "openai",
    providers: {
      openai: { method: "api_key", model: "model", credential_id: "c".repeat(32), credentials: { api_key: "secret" } },
    },
  };

  test("a v1 store without the new fields still loads", async () => {
    writeRaw(
      file,
      '{"default_provider": "openai", "providers": {"openai": {"credential_id": ' +
        '"0123456789abcdef0123456789abcdef", "credentials": {"api_key": "sk-old"}, ' +
        '"method": "api_key", "model": "gpt-4o"}}, "version": 1}\n',
    );
    const store = new AuthStore(file);
    const selection = store.selection();
    expect(selection?.provider).toBe("openai");
    expect(selection?.method).toBe("api_key");
    expect(selection?.endpoint).toBeNull();
    const credentials = (await store.credentials("openai", selection!.credentialId)) as ApiKeyCredentials;
    expect(credentials.apiKey).toBe("sk-old");
  });

  const subscription = {
    access_token: "a",
    refresh_token: "r",
    expires_at: 9000,
    account_id: "acct",
    residency: null,
  };
  const malformed: string[] = [
    "{",
    "[]",
    '{"version":99}',
    '{"secret":"do-not-echo"}',
    entryStore("p", { method: "none" }),
    entryStore("p", { credentials: null }),
    entryStore("p", { endpoint: ["do-not-echo"] }),
    entryStore("p", { secret: "do-not-echo" }),
    entryStore("p", { header_names: ["X-Gateway"] }),
    entryStore("openai", { organization: ["do-not-echo"] }),
    entryStore("openai", { organization: "" }),
    entryStore("openai", { project: "do not echo" }),
    entryStore("p", { organization: "do-not-echo" }),
    entryStore("openai", { organization: null }),
    entryStore("openai", { project: null }),
    entryStore("ollama", { credentials: { api_key: "k", headers: { X: "do-not-echo" } } }),
    entryStore("openai", { method: "subscription", project: "do-not-echo", credentials: subscription }),
    entryStore("p", { credentials: { api_key: "k", headers: [["X", "do-not-echo"]] } }),
    entryStore("p", { credentials: { api_key: "k", headers: {} } }),
    entryStore("p", { credentials: { api_key: "k", headers: { X: 1 } } }),
    entryStore("p", { credentials: { api_key: "k", headers: { Authorization: "do-not-echo" } } }),
    entryStore("p", { credentials: { api_key: "k", headers: { X: "do-not-echo\n" } } }),
    entryStore("p", { credentials: { api_key: "k", headers: { X: "do-not-echo", x: "v" } } }),
    entryStore("p", { credentials: { api_key: "k", secret: "do-not-echo" } }),
    entryStore("openai", { credentials: { api_key: "k", headers: { "OpenAI-Project": "do-not-echo" } } }),
    entryStore("p").replace('"version":1', '"version":1.0'),
    entryStore("p").replace('"version":1', '"version":1,"version":1'),
    entryStore("openai", { method: "subscription", credentials: { ...subscription, extra: "do-not-echo" } }),
    entryStore("openai", { method: "subscription", credentials: { ...subscription, account_id: undefined } }),
  ];
  for (const [index, contents] of malformed.entries()) {
    test(`a malformed store fails without echoing its payload: ${index} ${contents.slice(0, 50)}`, () => {
      writeRaw(file, contents);
      const error = thrown(() => new AuthStore(file).read());
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Credential storage is malformed; reconnect after repairing it");
      expect(error.message).not.toContain("do-not-echo");
      expect(error.cause).toBeUndefined();
    });
  }

  test("a store that isn't UTF-8 is malformed", () => {
    const text = new TextEncoder().encode(entryStore("p"));
    const at = text.indexOf("m".charCodeAt(0));
    text[at] = 0xff;
    writeRaw(file, text);
    expect(thrown(() => new AuthStore(file).read()).message).toBe(
      "Credential storage is malformed; reconnect after repairing it",
    );
  });

  test("a store reads a null endpoint", () => {
    writeRaw(file, entryStore("p", { endpoint: null }));
    expect(new AuthStore(file).selection("p")?.endpoint).toBeNull();
  });

  for (const target of ["directory", "file", "symlink"]) {
    test(`a store rejects an insecure path: ${target}`, () => {
      writeStore(file, savedStore);
      if (target === "directory") {
        fs.chmodSync(join(tmp, "auth"), 0o755);
      } else if (target === "file") {
        fs.chmodSync(file, 0o644);
      } else {
        const other = join(tmp, "auth", "other.json");
        fs.renameSync(file, other);
        fs.symlinkSync(other, file);
      }
      const error = thrown(() => new AuthStore(file).read());
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toMatch(/private|regular|symbolic/);
    });
  }

  test("a store whose directory is a symlink is refused", () => {
    writeStore(join(tmp, "real", "auth.json"), savedStore);
    fs.symlinkSync(join(tmp, "real"), join(tmp, "auth"));
    const error = thrown(() => new AuthStore(file).read());
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("Could not access private credential storage");
    expect(error.cause).toBeUndefined();
  });

  test("a store rejects a hard-linked file", () => {
    writeStore(file, savedStore);
    fs.linkSync(file, join(tmp, "auth", "alias"));
    expect(thrown(() => new AuthStore(file).read()).message).toBe(
      "Credential storage must use a regular private path",
    );
  });

  test("a store rejects a FIFO", () => {
    writeStore(file, savedStore);
    fs.unlinkSync(file);
    execFileSync("mkfifo", [file]);
    fs.chmodSync(file, 0o600);
    expect(thrown(() => new AuthStore(file).read()).message).toBe(
      "Credential storage must use a regular private path",
    );
  });

  test("a directory swapped during the file open is refused", () => {
    writeStore(file, savedStore);
    const directory = join(tmp, "auth");
    const real = fs.openSync;
    let swapped = false;
    const spy = spyOn(fs, "openSync").mockImplementation(((target: fs.PathLike, ...rest: unknown[]) => {
      if (target === file && !swapped) {
        swapped = true;
        fs.renameSync(directory, join(tmp, "moved"));
        writeStore(file, savedStore);
      }
      return Reflect.apply(real, fs, [target, ...rest]);
    }) as typeof fs.openSync);
    try {
      const error = thrown(() => new AuthStore(file).read());
      expect(swapped).toBe(true);
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Credential storage must use a regular private path");
    } finally {
      spy.mockRestore();
    }
    expect(new AuthStore(file).selection()?.provider).toBe("openai");
  });

  test("a directory swapped before a missing file's open is refused", () => {
    const directory = join(tmp, "auth");
    fs.mkdirSync(directory, { mode: 0o700 });
    const real = fs.openSync;
    let swapped = false;
    const spy = spyOn(fs, "openSync").mockImplementation(((target: fs.PathLike, ...rest: unknown[]) => {
      if (target === file && !swapped) {
        swapped = true;
        fs.renameSync(directory, join(tmp, "moved"));
        fs.mkdirSync(directory, { mode: 0o700 });
      }
      return Reflect.apply(real, fs, [target, ...rest]);
    }) as typeof fs.openSync);
    try {
      const error = thrown(() => new AuthStore(file).read());
      expect(swapped).toBe(true);
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Credential storage must use a regular private path");
    } finally {
      spy.mockRestore();
    }
  });

  test("a directory swapped before a write never receives the store file", async () => {
    writeStore(file, savedStore);
    const directory = join(tmp, "auth");
    const real = fs.readSync;
    let swapped = false;
    // The swap lands after the load's directory compare, before the write creates its file.
    const spy = spyOn(fs, "readSync").mockImplementation(((...args: unknown[]) => {
      if (!swapped) {
        swapped = true;
        fs.renameSync(directory, join(tmp, "moved"));
        fs.mkdirSync(directory, { mode: 0o700 });
      }
      return Reflect.apply(real, fs, args);
    }) as typeof fs.readSync);
    let error: Error;
    try {
      error = await rejected(new AuthStore(file).save("anthropic", new ApiKeyCredentials("new-key"), "new-model"));
    } finally {
      spy.mockRestore();
    }
    expect(swapped).toBe(true);
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("Credential storage must use a regular private path");
    expect(fs.readdirSync(directory).filter((name) => name === "auth.json" || name.startsWith(".auth-"))).toEqual(
      [],
    );
    expect(new AuthStore(join(tmp, "moved", "auth.json")).selection()?.provider).toBe("openai");
  });

  const timings: [string, Record<string, number>][] = [
    ["NaN lock timeout", { lockTimeoutMs: Number.NaN }],
    ["infinite lock timeout", { lockTimeoutMs: Number.POSITIVE_INFINITY }],
    ["zero lock timeout", { lockTimeoutMs: 0 }],
    ["NaN refresh margin", { refreshMarginMs: Number.NaN }],
    ["infinite refresh margin", { refreshMarginMs: Number.POSITIVE_INFINITY }],
    ["negative refresh margin", { refreshMarginMs: -1 }],
  ];
  for (const [label, options] of timings) {
    test(`a store rejects bad timings: ${label}`, () => {
      const error = thrown(() => new AuthStore(join(tmp, "auth.json"), options));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe("Invalid credential-store timing");
    });
  }

  test("the default path uses the composer environment only", () => {
    const env = (values: Record<string, string>) => (name: string) => values[name];
    const config = { XDG_CONFIG_HOME: join(tmp, "config"), CODEX_HOME: join(tmp, "native") };
    expect(new AuthStore(null, { env: env(config) }).path).toBe(join(tmp, "config", "yian", "auth.json"));
    expect(new AuthStore(undefined, { env: env({ ...config, YIAN_AUTH_FILE: "" }) }).path).toBe(
      join(tmp, "config", "yian", "auth.json"),
    );
    const override = join(tmp, "specific", "tokens.json");
    expect(new AuthStore(null, { env: env({ ...config, YIAN_AUTH_FILE: override }) }).path).toBe(override);
    const home = join(tmp, "home");
    expect(new AuthStore(null, { env: env({ HOME: home, XDG_CONFIG_HOME: "" }) }).path).toBe(
      join(home, ".config", "yian", "auth.json"),
    );
    expect(new AuthStore(null, { env: env({ HOME: home, YIAN_AUTH_FILE: "~/x/../tokens.json" }) }).path).toBe(
      join(home, "tokens.json"),
    );
    expect(new AuthStore("~", { env: env({ HOME: home }) }).path).toBe(home);
    expect(new AuthStore("~x", { env: env({ HOME: home }) }).path).toBe(join(process.cwd(), "~x"));
    expect(fs.readdirSync(tmp)).toEqual([]);
  });

  test("savedSecrets on a missing store is empty and creates nothing", () => {
    const missing = new AuthStore(join(tmp, "missing", "auth.json"));
    expect(missing.savedSecrets("anthropic")).toEqual([]);
    expect(missing.read()).toEqual({ defaultProvider: null, selections: [] });
    expect(missing.selection()).toBeNull();
    expect(fs.existsSync(join(tmp, "missing"))).toBe(false);
  });

  test("an aborted signal stops each read first", () => {
    writeStore(file, savedStore);
    const controller = new AbortController();
    const reason = new Error("stopped");
    controller.abort(reason);
    const store = new AuthStore(file, { signal: controller.signal });
    expect(thrown(() => store.read())).toBe(reason);
    expect(thrown(() => store.selection())).toBe(reason);
    expect(thrown(() => store.savedSecrets("openai"))).toBe(reason);
  });

  /** Return subscription credentials for account `accountId`. */
  function oauth(accessToken: string, refreshToken: string, expiresAt: number, accountId = "acct"): OAuthCredentials {
    return new OAuthCredentials({ accessToken, refreshToken, expiresAt, accountId });
  }

  /** A scheduler whose clock stays at `now` epoch milliseconds and whose sleeps are real. */
  function fixedClock(now: number): Scheduler {
    return { now: () => now, sleep: (ms, signal) => realScheduler.sleep(ms, signal) };
  }

  /** Return the permission bits of `path`. */
  function mode(path: string): number {
    return fs.statSync(path).mode & 0o777;
  }

  /** Save one login of each kind and return the selections by provider. */
  async function savedKinds(store: AuthStore): Promise<Record<string, ProviderSelection>> {
    await store.save("anthropic", new ApiKeyCredentials("k-anth"), "claude-x");
    await store.save("ollama", null, "llama3");
    await store.save("azure", new ApiKeyCredentials("k-az"), "dep", { endpoint: "https://r.openai.azure.com/" });
    await store.save("openai", oauth("access", "refresh", 9000), "gpt");
    return Object.fromEntries(store.read().selections.map((item) => [item.provider, item]));
  }

  /** An API key with two connection headers. */
  function gateway(key = "sk-open"): ApiKeyCredentials {
    return new ApiKeyCredentials(key, { "X-Gateway": "gw-secret", "A-Tag": "tag-secret" });
  }

  /** Write `source` as a script in the test's directory and return its path. */
  function script(name: string, source: string): string {
    const path = join(tmp, name);
    fs.writeFileSync(path, source);
    return path;
  }

  /** Start a child process that holds the lock on `lockPath`; resolve once it has it. */
  async function spawnHolder(lockPath: string): Promise<{ child: ChildProcess; closed: Promise<unknown> }> {
    const source =
      `import { lockFile } from ${JSON.stringify(PLATFORM_URL)};\n` +
      `const lock = lockFile(process.argv[2]);\n` +
      `console.log("locked");\n` +
      `for await (const _ of process.stdin);\n` +
      `lock.release();\n`;
    const child = spawn(process.execPath, [script("holder.ts", source), lockPath], { stdio: "pipe" });
    const closed = new Promise((resolve) => child.on("close", resolve));
    let stderr = "";
    child.stderr!.setEncoding("utf8").on("data", (chunk: string) => {
      stderr += chunk;
    });
    try {
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error(`the holder printed nothing for 5 s; stderr:\n${stderr}`)),
          5000,
        );
        child.stdout!.setEncoding("utf8").on("data", (chunk: string) => {
          if (chunk.includes("locked")) {
            clearTimeout(timer);
            resolve();
          }
        });
        child.on("error", (error) => {
          clearTimeout(timer);
          reject(error);
        });
        child.on("close", () => {
          clearTimeout(timer);
          reject(new Error(`the holder exited; stderr:\n${stderr}`));
        });
      });
    } catch (error) {
      child.kill("SIGKILL");
      throw error;
    }
    return { child, closed };
  }

  test("a read creates no files and save commits the selection", async () => {
    const path = join(tmp, "private", "auth.json");
    const store = new AuthStore(path);
    expect(store.read().selections).toEqual([]);
    expect(fs.existsSync(join(tmp, "private"))).toBe(false);
    const selection = await store.save("openai", new ApiKeyCredentials("secret-key"), "gpt-test");
    expect(selection.provider).toBe("openai");
    expect(selection.method).toBe("api_key");
    expect(selection.model).toBe("gpt-test");
    expect(selection.credentialId).toMatch(/^[0-9a-f]{32}$/);
    expect(store.read().defaultProvider).toBe("openai");
    expect(store.selection()).toEqual(selection);
    const credentials = (await store.credentials("openai", selection.credentialId)) as ApiKeyCredentials;
    expect(credentials.apiKey).toBe("secret-key");
    expect(mode(path)).toBe(0o600);
    expect(mode(join(tmp, "private"))).toBe(0o700);
    expect(mode(`${path}.lock`)).toBe(0o600);
    expect(inspect(store.read())).not.toContain("secret-key");
  });

  test("reconnection replaces the method and invalidates the previous identity", async () => {
    const store = new AuthStore(file);
    const first = await store.save("openai", new ApiKeyCredentials("old-secret"), "old-model");
    const second = await store.save("openai", oauth("access", "refresh", 9000), "new-model");
    expect(first.credentialId).not.toBe(second.credentialId);
    expect(store.selection()).toEqual(second);
    expect(fs.readFileSync(file, "utf8")).not.toContain("old-secret");
    const error = await rejected(store.credentials("openai", first.credentialId));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("The saved connection changed");
  });

  test("refresh is serialized and preserves identity", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    const refreshed: OAuthCredentials[] = [];
    const refresh = (credentials: OAuthCredentials) => {
      refreshed.push(credentials);
      return oauth("new", "new-refresh", 5000);
    };
    const results = await Promise.all(
      Array.from({ length: 12 }, () => store.credentials("openai", selection.credentialId, { refresh })),
    );
    expect(refreshed.length).toBe(1);
    expect(refreshed[0]!.accessToken).toBe("old");
    expect(results.every((result) => (result as OAuthCredentials).accessToken === "new")).toBe(true);
    expect(store.selection()).toEqual(selection);
  });

  /** Return a promise and the function that resolves it. */
  function gate(): [Promise<void>, () => void] {
    let open!: () => void;
    const opened = new Promise<void>((resolve) => {
      open = resolve;
    });
    return [opened, open];
  }

  /** Return a promise that rejects with `signal.reason` once `signal` aborts. */
  function aborted(signal: AbortSignal): Promise<never> {
    return new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason)));
  }

  /** Return the access token of each resolved `credentials` result. */
  function accessTokens(results: Credentials[]): string[] {
    return results.map((result) => (result as OAuthCredentials).accessToken);
  }

  test("a refresh slower than the lock timeout doesn't fail parallel callers", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const store = new AuthStore(file, { scheduler, lockTimeoutMs: 300 });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    let calls = 0;
    const refresh = async () => {
      calls++;
      await scheduler.sleep(1000);
      return oauth("new", "new-refresh", 5000);
    };
    const results = await Promise.all(
      Array.from({ length: 5 }, () => store.credentials("openai", selection.credentialId, { refresh })),
    );
    expect(accessTokens(results)).toEqual(Array(5).fill("new"));
    expect(calls).toBe(1);
  });

  test("a joiner whose signal aborts rejects with its reason and leaves the flight to the others", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    const [entered, enter] = gate();
    const [finished, finish] = gate();
    let calls = 0;
    const refresh = async () => {
      calls++;
      enter();
      await finished;
      return oauth("new", "new-refresh", 5000);
    };
    const controller = new AbortController();
    const reason = new Error("stopped");
    const leader = store.credentials("openai", selection.credentialId, { refresh });
    const joiner = new AuthStore(file, { scheduler: fixedClock(1_000_000), signal: controller.signal }).credentials(
      "openai",
      selection.credentialId,
      { refresh },
    );
    const others = Array.from({ length: 3 }, () => store.credentials("openai", selection.credentialId, { refresh }));
    await entered;
    controller.abort(reason);
    expect(await rejected(joiner)).toBe(reason);
    finish();
    expect(accessTokens(await Promise.all([leader, ...others]))).toEqual(Array(4).fill("new"));
    expect(calls).toBe(1);
  });

  test("a stopped leader rejects with its reason and one joiner refreshes for the rest", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    const controller = new AbortController();
    const reason = new Error("stopped");
    const [entered, enter] = gate();
    const calls: string[] = [];
    const leader = new AuthStore(file, { scheduler: fixedClock(1_000_000), signal: controller.signal }).credentials(
      "openai",
      selection.credentialId,
      {
        refresh: () => {
          calls.push("leader");
          enter();
          return aborted(controller.signal);
        },
      },
    );
    const refresh = async () => {
      calls.push("joiner");
      return oauth("new", "new-refresh", 5000);
    };
    const joiners = Array.from({ length: 4 }, () => store.credentials("openai", selection.credentialId, { refresh }));
    await entered;
    controller.abort(reason);
    expect(await rejected(leader)).toBe(reason);
    expect(accessTokens(await Promise.all(joiners))).toEqual(Array(4).fill("new"));
    expect(calls).toEqual(["leader", "joiner"]);
  });

  test("a failed refresh is shared, and a later call refreshes again", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    const failure = new Error("refresh failed");
    let calls = 0;
    const refresh = async () => {
      calls++;
      if (calls === 1) throw failure;
      return oauth("new", "new-refresh", 5000);
    };
    const results = await Promise.allSettled(
      Array.from({ length: 5 }, () => store.credentials("openai", selection.credentialId, { refresh })),
    );
    expect(results.every((result) => result.status === "rejected" && result.reason === failure)).toBe(true);
    expect(calls).toBe(1);
    const later = (await store.credentials("openai", selection.credentialId, { refresh })) as OAuthCredentials;
    expect(later.accessToken).toBe("new");
    expect(calls).toBe(2);
  });

  test("a flight is per store path, provider and credential id", async () => {
    const clock = fixedClock(1_000_000);
    const other = join(tmp, "other", "auth.json");
    const first = new AuthStore(file, { scheduler: clock });
    const second = new AuthStore(other, { scheduler: clock });
    // A copied store pins the same credential id on both paths, so only the path tells them apart.
    const copied = await first.save("openai", oauth("old", "old-refresh", 900), "model");
    writeRaw(other, fs.readFileSync(file));
    // Each refresh waits until both have started, so they finish only if they run in parallel.
    const [firstStarted, startFirst] = gate();
    const [secondStarted, startSecond] = gate();
    const both = Promise.all([firstStarted, secondStarted]);
    const parallel = await Promise.all([
      first.credentials("openai", copied.credentialId, {
        refresh: async () => {
          startFirst();
          await both;
          return oauth("first", "r", 5000);
        },
      }),
      second.credentials("openai", copied.credentialId, {
        refresh: async () => {
          startSecond();
          await both;
          return oauth("second", "r", 5000);
        },
      }),
    ]);
    expect(accessTokens(parallel)).toEqual(["first", "second"]);

    // Two providers on one path with the same credential id: each gets its own refresh, one after
    // the other under the lock.
    const path = join(tmp, "providers", "auth.json");
    const store = new AuthStore(path, { scheduler: clock });
    const a = await store.save("a", oauth("old", "r", 900), "model");
    await store.save("b", oauth("old", "r", 900), "model");
    const data = JSON.parse(fs.readFileSync(path, "utf8"));
    data.providers.b.credential_id = a.credentialId;
    writeRaw(path, JSON.stringify(data));
    const providers = await Promise.all([
      store.credentials("a", a.credentialId, { refresh: async () => oauth("new-a", "r", 5000) }),
      store.credentials("b", a.credentialId, { refresh: async () => oauth("new-b", "r", 5000) }),
    ]);
    expect(accessTokens(providers)).toEqual(["new-a", "new-b"]);

    // A caller pinned to another credential id of the same provider never receives this login.
    const original = await first.save("openai", oauth("old", "old-refresh", 900), "model");
    const [entered, enter] = gate();
    const [finished, finish] = gate();
    const pinnedFirst = first.credentials("openai", original.credentialId, {
      refresh: async () => {
        enter();
        await finished;
        return oauth("pinned", "r", 5000);
      },
    });
    await entered;
    // Replace the file while the first refresh holds the lock, so the next caller pins a different
    // credential id of the same provider (a shared flight would hand it `pinned`).
    const replacement = await second.save("openai", oauth("old", "old-refresh", 900), "model");
    writeRaw(file, fs.readFileSync(other));
    let replacementCalls = 0;
    const pinnedReplacement = first.credentials("openai", replacement.credentialId, {
      refresh: async () => {
        replacementCalls++;
        return oauth("replacement", "r", 5000);
      },
    });
    finish();
    expect(((await pinnedFirst) as OAuthCredentials).accessToken).toBe("pinned");
    expect((await rejected(pinnedReplacement)).message).toBe("The saved connection changed");
    expect(replacementCalls).toBe(0);
  });

  test("without a refresh, parallel callers each fail on an expired token", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    const errors = await Promise.all(
      Array.from({ length: 5 }, () => rejected(store.credentials("openai", selection.credentialId))),
    );
    expect(errors.map((error) => error.message)).toEqual(
      Array(5).fill("Subscription credentials expired; sign in again"),
    );
    expect(new Set(errors).size).toBe(5);
  });

  test("the store saves each credential kind", async () => {
    const store = new AuthStore(file);
    const saved = await savedKinds(store);
    expect(store.read().selections.length).toBe(4);
    expect(Object.fromEntries(Object.entries(saved).map(([name, item]) => [name, item.method]))).toEqual({
      anthropic: "api_key",
      ollama: "none",
      azure: "api_key",
      openai: "subscription",
    });
    expect(Object.fromEntries(Object.entries(saved).map(([name, item]) => [name, item.endpoint]))).toEqual({
      anthropic: null,
      ollama: null,
      azure: "https://r.openai.azure.com/",
      openai: null,
    });
    expect(saved["ollama"]!.model).toBe("llama3");
    expect(mode(file)).toBe(0o600);
    expect(inspect(store.read())).not.toContain("k-anth");
  });

  test("credentials for saved kinds", async () => {
    const store = new AuthStore(file);
    const saved = await savedKinds(store);
    const anthropic = (await store.credentials("anthropic", saved["anthropic"]!.credentialId)) as ApiKeyCredentials;
    expect(anthropic.apiKey).toBe("k-anth");
    const azure = (await store.credentials("azure", saved["azure"]!.credentialId)) as ApiKeyCredentials;
    expect(azure.apiKey).toBe("k-az");
    const error = await rejected(store.credentials("ollama", saved["ollama"]!.credentialId));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("ollama has no saved credential");
  });

  test("savedSecrets returns each kind without refresh", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(10_000_000) });
    await store.save("anthropic", new ApiKeyCredentials("k-anth"), "claude-x");
    await store.save("openai", oauth("access", "refresh", 900), "gpt");
    await store.save("ollama", null, "llama3");
    const snapshot = () => [fs.statSync(file, { bigint: true }).mtimeNs, fs.readFileSync(file)];
    const before = snapshot();
    expect(store.savedSecrets("anthropic")).toEqual(["k-anth"]);
    expect(store.savedSecrets("openai")).toEqual(["access", "refresh", "acct"]);
    expect(store.savedSecrets("ollama")).toEqual([]);
    expect(store.savedSecrets("google")).toEqual([]);
    expect(snapshot()).toEqual(before);
  });

  test("an open snapshot survives a concurrent atomic replacement", async () => {
    const store = new AuthStore(file);
    const original = await store.save("openai", new ApiKeyCredentials("first"), "first-model");
    const fd = fs.openSync(file, "r");
    const newer = await store.save("openai", new ApiKeyCredentials("second"), "second-model");
    const real = fs.openSync;
    let handedOver = false;
    const spy = spyOn(fs, "openSync").mockImplementation(((target: fs.PathLike, ...rest: unknown[]) => {
      if (target === file) {
        handedOver = true;
        return fd;
      }
      return Reflect.apply(real, fs, [target, ...rest]);
    }) as typeof fs.openSync);
    try {
      expect(fs.fstatSync(fd).nlink).toBe(0);
      expect(store.selection()).toEqual(original);
    } finally {
      spy.mockRestore();
      // The store closes the descriptor it was handed.
      if (!handedOver) fs.closeSync(fd);
    }
    expect(handedOver).toBe(true);
    expect(store.selection()).toEqual(newer);
  });

  test("remove drops a provider and the default naming it", async () => {
    const store = new AuthStore(file);
    await store.save("openai", new ApiKeyCredentials("k-open"), "gpt");
    await store.save("anthropic", new ApiKeyCredentials("k-anth"), "claude-x");
    expect(store.read().defaultProvider).toBe("anthropic");
    await store.remove("anthropic");
    let state = store.read();
    expect(state.defaultProvider).toBeNull();
    expect(state.selections.map((item) => item.provider)).toEqual(["openai"]);
    expect(fs.readFileSync(file, "utf8")).not.toContain("k-anth");
    expect((await rejected(store.remove("anthropic"))).message).toBe(
      "There is no saved anthropic login; run 'yian auth' to list them",
    );

    await store.save("ollama", null, "llama3", { makeDefault: false });
    await store.save("azure", new ApiKeyCredentials("k-az"), "dep", { endpoint: "https://r.openai.azure.com/" });
    await store.remove("ollama");
    await store.remove("azure");
    state = store.read();
    expect(state.defaultProvider).toBeNull();
    expect(state.selections.map((item) => item.provider)).toEqual(["openai"]);

    const missing = new AuthStore(join(tmp, "missing", "auth.json"));
    const error = await rejected(missing.remove("openai"));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("There is no saved openai login; run 'yian auth' to list them");
    expect(fs.existsSync(join(tmp, "missing"))).toBe(false);

    const legacy = join(tmp, "legacy", "auth.json");
    const entry = {
      method: "api_key",
      model: "m",
      credential_id: "c".repeat(32),
      credentials: { api_key: "k-legacy" },
    };
    writeStore(legacy, { version: 1, default_provider: "legacy-x", providers: { "legacy-x": entry, openai: entry } });
    await new AuthStore(legacy).remove("legacy-x");
    state = new AuthStore(legacy).read();
    expect(state.defaultProvider).toBeNull();
    expect(state.selections.map((item) => item.provider)).toEqual(["openai"]);
  });

  test("setDefault requires a saved provider", async () => {
    const store = new AuthStore(file);
    const first = await store.save("openai", new ApiKeyCredentials("k-open"), "gpt");
    const second = await store.save("anthropic", new ApiKeyCredentials("k-anth"), "claude-x");
    await store.setDefault("openai");
    expect(store.read().defaultProvider).toBe("openai");
    expect((await rejected(store.setDefault("google"))).message).toBe(
      "There is no saved google login; run 'yian auth' to list them",
    );
    expect(store.read().defaultProvider).toBe("openai");
    expect(store.selection("openai")).toEqual(first);
    expect(store.selection("anthropic")).toEqual(second);

    await store.save("ollama", null, "llama3", { makeDefault: false });
    await store.save("azure", new ApiKeyCredentials("k-az"), "dep", {
      endpoint: "https://r.openai.azure.com/",
      makeDefault: false,
    });
    await store.setDefault("ollama");
    expect(store.selection()?.provider).toBe("ollama");
    await store.setDefault("azure");
    expect(store.selection()?.endpoint).toBe("https://r.openai.azure.com/");

    const missing = new AuthStore(join(tmp, "missing", "auth.json"));
    expect((await rejected(missing.setDefault("openai"))).message).toBe(
      "There is no saved openai login; run 'yian auth' to list them",
    );
    expect(fs.existsSync(join(tmp, "missing"))).toBe(false);
  });

  test("delete removes the saved store and keeps its lock file", async () => {
    const store = new AuthStore(file);
    await store.save("openai", new ApiKeyCredentials("k-open"), "gpt");
    await store.delete();
    expect(fs.existsSync(file)).toBe(false);
    expect(fs.existsSync(`${file}.lock`)).toBe(true);
    expect(store.read().selections).toEqual([]);
    await store.delete();
    expect(fs.readdirSync(join(tmp, "auth"))).toEqual(["auth.json.lock"]);
  });

  test("delete waits for a held lock, then removes the store and keeps its lock file", async () => {
    await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
    const scheduler = virtualScheduler(1_000_000);
    const holder = lockFile(`${file}.lock`);
    // Checked at +200, while the holder still has the lock; released at +250, between two of
    // waitForLock's 100 ms polls, so the delete takes the lock on the +300 poll.
    let stillThere = false;
    void scheduler.sleep(200).then(() => {
      stillThere = fs.existsSync(file);
    });
    void scheduler.sleep(250).then(() => holder.release());
    await new AuthStore(file, { scheduler, lockTimeoutMs: 1_000 }).delete();
    expect(stillThere).toBe(true);
    expect(scheduler.now()).toBe(1_000_300);
    expect(fs.existsSync(file)).toBe(false);
    expect(fs.existsSync(`${file}.lock`)).toBe(true);
  });

  test("delete waits for a refresh in flight and the store stays deleted", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const store = new AuthStore(file, { scheduler });
    const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
    const [entered, enter] = gate();
    const [finished, finish] = gate();
    const refreshed = store.credentials("openai", selection.credentialId, {
      refresh: async () => {
        enter();
        await finished;
        return oauth("new", "new-refresh", 5000);
      },
    });
    await entered;
    let deleted = false;
    const deleting = store.delete().then(() => {
      deleted = true;
    });
    // Past waitForLock's +100 and +200 polls, the delete must still wait on the refresh's lock.
    await scheduler.sleep(250);
    expect(deleted).toBe(false);
    finish();
    expect(((await refreshed) as OAuthCredentials).accessToken).toBe("new");
    await deleting;
    expect(fs.existsSync(file)).toBe(false);
    expect(fs.existsSync(`${file}.lock`)).toBe(true);
    expect(store.read().selections).toEqual([]);
  });

  test("delete without a store or lock file creates nothing", async () => {
    const missing = join(tmp, "missing");
    await new AuthStore(join(missing, "auth.json")).delete();
    expect(fs.existsSync(missing)).toBe(false);

    const empty = join(tmp, "empty");
    fs.mkdirSync(empty, { mode: 0o700 });
    await new AuthStore(join(empty, "auth.json")).delete();
    expect(fs.readdirSync(empty)).toEqual([]);
  });

  test("delete removes a store without its lock file", async () => {
    await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
    fs.unlinkSync(`${file}.lock`);
    await new AuthStore(file).delete();
    expect(fs.existsSync(file)).toBe(false);
    expect(fs.existsSync(`${file}.lock`)).toBe(true);
  });

  test("delete removes a symlink without following it", async () => {
    await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
    const other = join(tmp, "auth", "other.json");
    fs.renameSync(file, other);
    fs.symlinkSync(other, file);
    await new AuthStore(file).delete();
    expect(() => fs.lstatSync(file)).toThrow(expect.objectContaining({ code: "ENOENT" }));
    expect(fs.existsSync(other)).toBe(true);
  });

  for (const saved of [false, true]) {
    test(`delete refuses a directory others can read: ${saved ? "saved" : "empty"}`, async () => {
      if (saved) {
        await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
      } else {
        fs.mkdirSync(join(tmp, "auth"), { mode: 0o700 });
      }
      fs.chmodSync(join(tmp, "auth"), 0o755);
      const error = await rejected(new AuthStore(file).delete());
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Credential storage must be private to its owner");
      expect(fs.existsSync(file)).toBe(saved);
    });
  }

  test("an aborted signal stops delete before it touches storage", async () => {
    await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
    const controller = new AbortController();
    const reason = new Error("stopped");
    controller.abort(reason);
    expect(await rejected(new AuthStore(file, { signal: controller.signal }).delete())).toBe(reason);
    expect(fs.existsSync(file)).toBe(true);
  });

  test("the store saves connection settings with a login", async () => {
    const store = new AuthStore(file);
    const selection = await store.save("openai", gateway(), "gpt", { organization: "org-1", project: "proj-1" });
    expect([selection.organization, selection.project]).toEqual(["org-1", "proj-1"]);
    expect(selection.headerNames).toEqual(["A-Tag", "X-Gateway"]);
    expect(store.selection("openai")).toEqual(selection);
    const credentials = (await store.credentials("openai", selection.credentialId)) as ApiKeyCredentials;
    expect(credentials.apiKey).toBe("sk-open");
    expect(credentials.headers).toEqual([
      ["A-Tag", "tag-secret"],
      ["X-Gateway", "gw-secret"],
    ]);
    let entry = JSON.parse(fs.readFileSync(file, "utf8"))["providers"]["openai"];
    expect([entry.organization, entry.project]).toEqual(["org-1", "proj-1"]);
    expect(entry.credentials).toEqual({
      api_key: "sk-open",
      headers: { "A-Tag": "tag-secret", "X-Gateway": "gw-secret" },
    });

    const anthropic = await store.save("anthropic", new ApiKeyCredentials("k", { "X-G": "v" }), "c");
    expect(anthropic.headerNames).toEqual(["X-G"]);
    expect([anthropic.organization, anthropic.project]).toEqual([null, null]);
    expect(store.selection("openai")).toEqual(selection);

    const plain = await store.save("openai", new ApiKeyCredentials("sk-open"), "gpt");
    expect([plain.organization, plain.project, plain.headerNames]).toEqual([null, null, []]);
    expect(((await store.credentials("openai", plain.credentialId)) as ApiKeyCredentials).headers).toEqual([]);
    entry = JSON.parse(fs.readFileSync(file, "utf8"))["providers"]["openai"];
    expect(Object.keys(entry)).toEqual(["credential_id", "credentials", "method", "model"]);
    expect(entry.credentials).toEqual({ api_key: "sk-open" });
  });

  test("save refuses a login its provider can't use and leaves the store unchanged", async () => {
    const store = new AuthStore(file);
    await store.save("openai", new ApiKeyCredentials("k-open"), "gpt");
    const before = fs.readFileSync(file, "utf8");
    const cases: [string, ApiKeyCredentials | OAuthCredentials | null, { endpoint?: string }][] = [
      ["anthropic", new ApiKeyCredentials("k"), { endpoint: "https://r.openai.azure.com/" }],
      ["azure", new ApiKeyCredentials("k"), {}],
      ["azure", new ApiKeyCredentials("k"), { endpoint: "http://r.openai.azure.com/" }],
      ["ollama", new ApiKeyCredentials("k"), {}],
      ["openai", null, {}],
      ["anthropic", null, {}],
      ["anthropic", oauth("access", "refresh", 900), {}],
    ];
    for (const [provider, credentials, options] of cases) {
      const error = await rejected(store.save(provider, credentials, "model", options));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(`A ${provider} login can't be saved with this credential or endpoint`);
    }
    expect(fs.readFileSync(file, "utf8")).toBe(before);
    await store.save("vllm", null, "m", { makeDefault: false });
    await store.save("azure", new ApiKeyCredentials("k-az"), "dep", { endpoint: "https://r.openai.azure.com/" });
  });

  test("a store without connection settings rewrites byte-identically", async () => {
    const raw =
      '{"default_provider": "openai", "providers": {"anthropic": {"credential_id": ' +
      '"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "credentials": {"api_key": "k-anth"}, "method": "api_key", ' +
      '"model": "claude-x"}, "azure": {"credential_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "credentials": ' +
      '{"api_key": "k-az"}, "endpoint": "https://r.openai.azure.com/", "method": "api_key", "model": "dep"}, ' +
      '"ollama": {"credential_id": "cccccccccccccccccccccccccccccccc", "credentials": null, "method": "none", ' +
      '"model": "llama3"}, "openai": {"credential_id": "dddddddddddddddddddddddddddddddd", "credentials": ' +
      '{"access_token": "access", "account_id": "acct", "expires_at": 9000, "refresh_token": "refresh", ' +
      '"residency": null}, "method": "subscription", "model": "gpt"}}, "version": 1}\n';
    writeRaw(file, raw);
    const store = new AuthStore(file);
    await store.setDefault("anthropic");
    await store.setDefault("openai");
    expect(fs.readFileSync(file, "utf8")).toBe(raw);
    for (const item of store.read().selections) {
      expect([item.organization, item.project, item.headerNames]).toEqual([null, null, []]);
    }
  });

  test("save writes the store as sorted, spaced JSON", async () => {
    const store = new AuthStore(file);
    const subscription = await store.save("openai", oauth("access", "refresh", 9000), "gpt");
    const key = await store.save("p", gateway(), "m", { endpoint: "https://e.example/", makeDefault: false });
    const text = fs.readFileSync(file, "utf8");
    expect(text).toBe(
      '{"default_provider": "openai", "providers": {"openai": {"credential_id": ' +
        `"${subscription.credentialId}", "credentials": {"access_token": "access", "account_id": "acct", ` +
        '"expires_at": 9000, "refresh_token": "refresh", "residency": null}, "method": "subscription", ' +
        `"model": "gpt"}, "p": {"credential_id": "${key.credentialId}", "credentials": {"api_key": "sk-open", ` +
        '"headers": {"A-Tag": "tag-secret", "X-Gateway": "gw-secret"}}, "endpoint": "https://e.example/", ' +
        '"method": "api_key", "model": "m"}}, "version": 1}\n',
    );
    const value = parseJson(text, "plain") as Record<string, unknown>;
    expect(value["version"]).toBe(1n);
    expect(value["default_provider"]).toBe("openai");
    expect(Object.keys(value["providers"] as object)).toEqual(["openai", "p"]);
  });

  test("connection settings values stay out of inspection", async () => {
    const store = new AuthStore(file);
    const selection = await store.save("openai", gateway(), "gpt", { organization: "org-1" });
    const credentials = await store.credentials("openai", selection.credentialId);
    for (const text of [inspect(credentials), inspect(selection), inspect(store.read())]) {
      expect(text).not.toContain("gw-secret");
      expect(text).not.toContain("tag-secret");
      expect(text).not.toContain("sk-open");
    }
    expect(inspect(selection)).toContain("X-Gateway");
  });

  test("savedSecrets lists header values", async () => {
    const store = new AuthStore(file);
    await store.save("openai", gateway(), "gpt");
    expect(store.savedSecrets("openai")).toEqual(["sk-open", "tag-secret", "gw-secret"]);
  });

  const refusedSettings: [string, () => ApiKeyCredentials | OAuthCredentials | null, Record<string, string>][] = [
    ["anthropic", () => new ApiKeyCredentials("k"), { organization: "org-1" }],
    ["azure", () => new ApiKeyCredentials("k"), { project: "proj-1" }],
    ["ollama", () => null, { organization: "org-1" }],
    ["openai", () => oauth("a", "r", 9000), { project: "proj-1" }],
    ["openai", () => new ApiKeyCredentials("k"), { organization: "has space" }],
    ["openai", () => new ApiKeyCredentials("k"), { project: "" }],
    ["openai", () => new ApiKeyCredentials("k"), { organization: "o".repeat(257) }],
    ["openai", () => new ApiKeyCredentials("k", { "OpenAI-Organization": "o" }), {}],
  ];
  for (const [index, [provider, credentials, settings]] of refusedSettings.entries()) {
    test(`the store refuses connection settings a login cannot send: ${index} ${provider}`, async () => {
      const store = new AuthStore(file);
      const error = await rejected(store.save(provider, credentials(), "m", settings));
      expect(error).toBeInstanceOf(AuthError);
      for (const value of Object.values(settings)) {
        if (value) expect(error.message).not.toContain(value);
      }
      expect(fs.existsSync(join(tmp, "auth"))).toBe(false);
    });
  }

  test("save refuses a credential that isn't exactly a credential class", async () => {
    class Subclass extends ApiKeyCredentials {}
    const store = new AuthStore(file);
    for (const credentials of [new Subclass("k"), { method: "api_key", apiKey: "k", headers: [] }, undefined]) {
      const error = await rejected(store.save("p", credentials as unknown as ApiKeyCredentials, "m"));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Unsupported credential type");
    }
    expect(fs.existsSync(join(tmp, "auth"))).toBe(false);
  });

  test("the store refuses a lock file that isn't private", async () => {
    const store = new AuthStore(file);
    const selection = await store.save("openai", new ApiKeyCredentials("secret"), "model");
    fs.chmodSync(`${file}.lock`, 0o644);
    const error = await rejected(store.save("openai", new ApiKeyCredentials("new"), "model"));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("Credential storage must be private to its owner");
    expect(store.selection()).toEqual(selection);
  });

  const lockedOperations: [string, (store: AuthStore) => Promise<unknown>][] = [
    ["save", (store) => store.save("openai", new ApiKeyCredentials("new"), "model-new")],
    ["delete", (store) => store.delete()],
  ];
  for (const [name, operation] of lockedOperations) for (const cancel of [false, true]) {
    test(`the lock wait is bounded and cancellable: ${name} ${cancel ? "cancel" : "timeout"}`, async () => {
      await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
      const scheduler = virtualScheduler(1_000_000);
      const controller = new AbortController();
      const reason = new Error("stopped");
      if (cancel) void scheduler.sleep(200).then(() => controller.abort(reason));
      const holder = lockFile(`${file}.lock`);
      try {
        const store = new AuthStore(file, { scheduler, signal: controller.signal, lockTimeoutMs: 500 });
        const error = await rejected(operation(store));
        if (cancel) {
          expect(error).toBe(reason);
          expect(scheduler.now()).toBe(1_000_200);
        } else {
          expect(error).toBeInstanceOf(AuthError);
          expect(error.message).toBe("Timed out waiting for credential storage");
          expect(scheduler.now()).toBe(1_000_500);
        }
      } finally {
        holder.release();
      }
      expect(new AuthStore(file).selection()?.model).toBe("model");
    });
  }

  test("an atomic write failure keeps the previous credentials and default", async () => {
    const store = new AuthStore(file);
    const before = await store.save("openai", new ApiKeyCredentials("old-key"), "old-model");
    const raw = fs.readFileSync(file);
    const spy = spyOn(fs, "renameSync").mockImplementation(() => {
      throw Object.assign(new Error("secret-value"), { code: "EIO" });
    });
    let error: Error;
    try {
      error = await rejected(store.save("anthropic", new ApiKeyCredentials("new-key"), "new-model"));
    } finally {
      spy.mockRestore();
    }
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("Could not access private credential storage");
    expect(error.cause).toBeUndefined();
    expect(fs.readFileSync(file)).toEqual(raw);
    expect(store.selection()).toEqual(before);
    expect(fs.readdirSync(join(tmp, "auth")).filter((name) => name.startsWith(".auth-"))).toEqual([]);
  });

  test("a lock failure is not ignored", async () => {
    fs.mkdirSync(join(tmp, "auth"), { mode: 0o700 });
    fs.mkdirSync(`${file}.lock`);
    const error = await rejected(new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model"));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("Credential storage must use a regular private path");
    expect(fs.existsSync(file)).toBe(false);
  });

  test("a failure of the lock itself is the access error", async () => {
    const access = (error: Error) => {
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Could not access private credential storage");
      expect(error.cause).toBeUndefined();
    };
    fs.mkdirSync(join(tmp, "auth"), { mode: 0o700 });
    // A private regular file that is not a database: the lock mechanism fails, not the lock check.
    fs.writeFileSync(`${file}.lock`, "not a database", { mode: 0o600 });
    access(await rejected(new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model")));
    expect(fs.existsSync(file)).toBe(false);

    fs.rmSync(`${file}.lock`);
    const store = new AuthStore(file);
    await store.save("openai", new ApiKeyCredentials("key"), "model", { makeDefault: false });
    await store.save("anthropic", new ApiKeyCredentials("other"), "model");
    const raw = fs.readFileSync(file);
    fs.rmSync(`${file}.lock`);
    fs.writeFileSync(`${file}.lock`, "not a database", { mode: 0o600 });
    access(await rejected(store.save("openai", new ApiKeyCredentials("new"), "model")));
    access(await rejected(store.setDefault("openai")));
    expect(fs.readFileSync(file)).toEqual(raw);
  });

  test("a changed connection prevents refresh and an account switch prevents the save", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const first = await store.save("openai", oauth("old", "refresh", 900), "model");
    const second = await store.save("openai", oauth("new", "refresh", 900), "model");
    const calls: OAuthCredentials[] = [];
    const refresh = (credentials: OAuthCredentials) => {
      calls.push(credentials);
      return oauth("switched", "other-refresh", 5000, "other-account");
    };
    expect((await rejected(store.credentials("openai", first.credentialId, { refresh }))).message).toBe(
      "The saved connection changed",
    );
    expect(calls).toEqual([]);
    const raw = fs.readFileSync(file);
    expect((await rejected(store.credentials("openai", second.credentialId, { refresh }))).message).toBe(
      "Subscription refresh changed account identity",
    );
    expect(fs.readFileSync(file)).toEqual(raw);
  });

  test("a refresh is checked: absent, expired, or not exactly OAuthCredentials", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "refresh", 900), "model");
    const raw = fs.readFileSync(file);
    const resolve = (refresh?: () => OAuthCredentials) =>
      rejected(store.credentials("openai", selection.credentialId, refresh && { refresh }));
    expect((await resolve()).message).toBe("Subscription credentials expired; sign in again");
    expect((await resolve(() => oauth("new", "refresh", 1000))).message).toBe(
      "Subscription refresh returned expired credentials",
    );
    class Subclass extends OAuthCredentials {}
    const subclass = new Subclass({ accessToken: "new", refreshToken: "r", expiresAt: 5000, accountId: "acct" });
    expect((await resolve(() => subclass)).message).toBe("Subscription refresh changed account identity");
    expect(fs.readFileSync(file)).toEqual(raw);
  });

  test("a refresh's own error propagates unchanged", async () => {
    const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
    const selection = await store.save("openai", oauth("old", "refresh", 900), "model");
    const failure = Object.assign(new Error("x"), { code: "EIO" });
    const refresh = () => {
      throw failure;
    };
    expect(await rejected(store.credentials("openai", selection.credentialId, { refresh }))).toBe(failure);
  });

  test("cancellation comes before file creation", async () => {
    const controller = new AbortController();
    const reason = new Error("stopped");
    controller.abort(reason);
    const path = join(tmp, "missing", "auth.json");
    const store = new AuthStore(path, { signal: controller.signal });
    expect(await rejected(store.save("openai", new ApiKeyCredentials("key"), "model"))).toBe(reason);
    expect(fs.existsSync(join(tmp, "missing"))).toBe(false);
  });

  test(
    "the lock holds across processes while a call in this process polls for it",
    async () => {
      const store = new AuthStore(file, { scheduler: fixedClock(1_000_000) });
      const selection = await store.save("openai", oauth("old", "old-refresh", 900), "model");
      let entered!: () => void;
      const inside = new Promise<void>((resolve) => {
        entered = resolve;
      });
      let finish!: () => void;
      const finished = new Promise<void>((resolve) => {
        finish = resolve;
      });
      const resolving = store.credentials("openai", selection.credentialId, {
        refresh: async () => {
          entered();
          await finished;
          return oauth("new", "new-refresh", 5000);
        },
      });
      await inside;
      const saving = store.save("anthropic", new ApiKeyCredentials("k-anth"), "claude-x", { makeDefault: false });
      // Each poll opens and closes a connection on the lock file; none may drop the held lock.
      await realScheduler.sleep(250);
      const probe = script(
        "probe.ts",
        `import { isLocked } from ${JSON.stringify(PLATFORM_URL)};\n` +
          `console.log(JSON.stringify(isLocked(process.argv[2])));\n`,
      );
      const result = spawnSync(process.execPath, [probe, `${file}.lock`], { encoding: "utf8", timeout: 10_000 });
      finish();
      const [credentials] = await Promise.all([resolving, saving]);
      expect(result.stdout.trim()).toBe("true");
      expect((credentials as OAuthCredentials).accessToken).toBe("new");
      const state = store.read();
      expect(state.defaultProvider).toBe("openai");
      expect(state.selections.map((item) => item.provider)).toEqual(["anthropic", "openai"]);
    },
    20_000,
  );

  test(
    "a lock held by another process makes save time out until that process exits",
    async () => {
      await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
      const { child, closed } = await spawnHolder(`${file}.lock`);
      try {
        const store = new AuthStore(file, { lockTimeoutMs: 300 });
        const error = await rejected(store.save("openai", new ApiKeyCredentials("new"), "model-new"));
        expect(error.message).toBe("Timed out waiting for credential storage");
        child.kill("SIGKILL");
        await closed;
        await store.save("openai", new ApiKeyCredentials("new"), "model-new");
        expect(store.selection()?.model).toBe("model-new");
      } finally {
        child.kill("SIGKILL");
      }
    },
    20_000,
  );
});

/** A request a local server received. */
interface Received {
  method: string;
  headers: http.IncomingHttpHeaders;
  body: string;
}

/** A local HTTP server on 127.0.0.1. */
interface LocalServer {
  url: string;
  received: Received[];
  close(): Promise<void>;
}

/** Start a server on 127.0.0.1 that records each request, then lets `handle` answer it. */
async function localServer(handle: (res: http.ServerResponse, received: Received) => void): Promise<LocalServer> {
  const received: Received[] = [];
  const server = http.createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk: string) => (body += chunk));
    req.on("end", () => {
      const each = { method: req.method!, headers: req.headers, body };
      received.push(each);
      handle(res, each);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/`,
    received,
    async close() {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

describe("request", () => {
  for (const status of [200, 401, 302]) {
    test(`builtin http decodes responses and never follows redirects: ${status}`, async () => {
      const body = '{"data":[{"id":"gpt-test"}]}';
      const server = await localServer((res) => {
        res.writeHead(status, { location: `${server.url}redirected`, "content-length": body.length });
        res.end(body);
      });
      try {
        const response = await request("GET", server.url, { headers: {}, timeoutMs: 1000 });
        expect(response).toBeInstanceOf(HttpResponse);
        expect(response.status).toBe(status);
        // A 3xx body is cancelled unread, and no caller reads a non-2xx body.
        expect(response.data).toEqual(status === 302 ? null : { data: [{ id: "gpt-test" }] });
        expect(server.received).toHaveLength(1);
        expect(server.received[0]!.method).toBe("GET");
      } finally {
        await server.close();
      }
    });
  }

  test("integers in data are bigints", async () => {
    const server = await localServer((res) => {
      res.writeHead(200);
      res.end('{"expires_in": 3600, "ratio": 1.5}');
    });
    try {
      const response = await request("GET", server.url, { headers: {}, timeoutMs: 1000 });
      expect(response.data).toEqual({ expires_in: 3600n, ratio: 1.5 });
    } finally {
      await server.close();
    }
  });

  test("a non-JSON body gives null data and keeps the status", async () => {
    const server = await localServer((res) => {
      res.writeHead(500);
      res.end("<html>oops</html>");
    });
    try {
      const response = await request("GET", server.url, { headers: {}, timeoutMs: 1000 });
      expect(response.status).toBe(500);
      expect(response.data).toBeNull();
    } finally {
      await server.close();
    }
  });

  for (const [bodyLimit, fits] of [
    [1024, false],
    [4096, true],
  ] as const) {
    test(`builtin http body limit raises a protocol error: ${bodyLimit}`, async () => {
      const body = JSON.stringify({ pad: "x".repeat(2048) });
      const server = await localServer((res) => res.end(body));
      try {
        const sent = request("GET", server.url, { headers: {}, timeoutMs: 2000, bodyLimit });
        if (fits) {
          expect((await sent).data).toEqual({ pad: "x".repeat(2048) });
        } else {
          const error = await rejected(sent);
          expect(error).toBeInstanceOf(ProviderError);
          expect((error as ProviderError).kind).toBe("protocol");
          expect(error.message).toBe("HTTP response is larger than 1024 bytes");
        }
      } finally {
        await server.close();
      }
    });
  }

  test("jsonBody is sent as JSON and form as urlencoded text", async () => {
    const server = await localServer((res, received) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ method: received.method, type: received.headers["content-type"], body: received.body }));
    });
    try {
      const json = await request("POST", server.url, {
        headers: { "Content-Type": "application/json" },
        timeoutMs: 1000,
        jsonBody: { client_id: "app", scope: "a b" },
      });
      expect(json.data).toEqual({
        method: "POST",
        type: "application/json",
        body: '{"client_id":"app","scope":"a b"}',
      });
      const form = await request("POST", server.url, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        timeoutMs: 1000,
        form: { grant_type: "refresh_token", scope: "a b&c" },
      });
      expect(form.data).toEqual({
        method: "POST",
        type: "application/x-www-form-urlencoded",
        body: "grant_type=refresh_token&scope=a+b%26c",
      });
    } finally {
      await server.close();
    }
  });

  test("builtin http deadline raises a timeout error", async () => {
    const server = await localServer(() => {});
    try {
      const start = Date.now();
      const error = await rejected(
        request("GET", server.url, { headers: {}, timeoutMs: 100, scheduler: realScheduler }),
      );
      expect(error).toBeInstanceOf(ProviderError);
      expect((error as ProviderError).kind).toBe("timeout");
      expect(Date.now() - start).toBeLessThan(1000);
    } finally {
      await server.close();
    }
  });

  test("a caller abort mid-request rejects with the abort reason", async () => {
    const controller = new AbortController();
    const reason = new Error("stopped by host");
    const server = await localServer(() => controller.abort(reason));
    try {
      const error = await rejected(
        request("GET", server.url, { headers: {}, timeoutMs: 5000, signal: controller.signal }),
      );
      expect(error).toBe(reason);
      expect(server.received).toHaveLength(1);
    } finally {
      await server.close();
    }
  });

  test("HttpResponse keeps its data out of inspection", () => {
    const response = new HttpResponse(200, { access_token: "secret-token" });
    expect(response.status).toBe(200);
    expect(response.data).toEqual({ access_token: "secret-token" });
    expect(Object.isFrozen(response)).toBe(true);
    for (const text of [JSON.stringify(response), String(response), inspect(response)]) {
      expect(text).not.toContain("secret-token");
    }
    expect(JSON.stringify(response)).toBe('{"status":200}');
  });
});

/** An unsigned JWT whose payload names `account` in OpenAI's auth claim, plus `claims`. */
function jwt(account: string | null = "account-test", claims: Record<string, unknown> = {}): string {
  const value = { "https://api.openai.com/auth": { chatgpt_account_id: account }, ...claims };
  return `header.${Buffer.from(JSON.stringify(value)).toString("base64url")}.signature`;
}

/** A token endpoint payload for `account`, with integers as `bigint`s as `request` decodes them. */
function tokens(account = "account-test", overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return { access_token: jwt(account), refresh_token: "refresh-secret", expires_in: 3600n, ...overrides };
}

/** A fake `AuthRequest` that records each call and replies from a queue. */
class FakeHttp {
  readonly calls: [string, string, Parameters<AuthRequest>[2]][] = [];
  /** The clock reading at each call, when `clock` is set. */
  readonly times: number[] = [];
  clock: Scheduler | undefined;
  readonly #responses: unknown[];

  constructor(...responses: unknown[]) {
    this.#responses = responses;
  }

  readonly request: AuthRequest = async (method, url, options) => {
    this.calls.push([method, url, options]);
    if (this.clock) this.times.push(this.clock.now());
    return this.#responses.shift() as HttpResponse;
  };
}

/** Return a never-settling promise, like a host that never answers. */
function never(): Promise<never> {
  return new Promise(() => {});
}

/** An `awaitCallback` answering with the state sent in `urls[0]`, then `&` and `query`. */
function callbackWith(urls: string[], query: string): (redirectUri: string) => string {
  return (redirectUri) => redirectUri + "?state=" + new URL(urls[0]!).searchParams.get("state") + "&" + query;
}

const GATEWAY = "gw-SENTINEL-token-9876";
const ORG = "org-SENTINEL-1234";
const PROJECT = "proj_SENTINEL-5678";

/** Check the error and its cause chain for every setting value. */
function assertNoSettings(error: unknown): void {
  for (let link = error; link !== undefined; link = (link as Error).cause) {
    const texts = [String(link), (link as Error).message, (link as Error).stack ?? "", inspect(link)];
    for (const value of [GATEWAY, ORG, PROJECT]) {
      for (const text of texts) expect(text).not.toContain(value);
    }
  }
}

const OLD = () =>
  new OAuthCredentials({
    accessToken: "old",
    refreshToken: "old-refresh",
    expiresAt: 1000,
    accountId: "acct",
    residency: "eu",
  });
const SUBSCRIPTION = () =>
  new OAuthCredentials({ accessToken: "access", refreshToken: "refresh", expiresAt: 9000, accountId: "acct" });

describe("OpenAIAuth", () => {
  test("device protocol uses JSON polls and server PKCE", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const http = new FakeHttp(
      new HttpResponse(200, { device_auth_id: "device-id", user_code: "ABC-DEF", interval: "2" }),
      new HttpResponse(403, { error: "pending" }),
      new HttpResponse(404, {}),
      new HttpResponse(200, { authorization_code: "auth-code", code_verifier: "server-verifier" }),
      new HttpResponse(200, tokens()),
    );
    const display: [string, string][] = [];
    const auth = new OpenAIAuth({ request: http.request, scheduler });
    const credentials = await auth.deviceLogin((url, code) => void display.push([url, code]));
    expect(display).toEqual([["https://auth.openai.com/codex/device", "ABC-DEF"]]);
    expect(credentials.accountId).toBe("account-test");
    expect(credentials.expiresAt).toBe(scheduler.now() / 1000 + 3600);
    expect(scheduler.now()).toBe(1_004_000);
    expect(http.calls[0]![2].jsonBody).toEqual({ client_id: "app_EMoamEEZ73f0CkXaXp7hrann" });
    expect(http.calls[1]![2].jsonBody).toEqual({ device_auth_id: "device-id", user_code: "ABC-DEF" });
    expect(http.calls.at(-1)![2].form).toEqual({
      grant_type: "authorization_code",
      code: "auth-code",
      code_verifier: "server-verifier",
      client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
      redirect_uri: "https://auth.openai.com/deviceauth/callback",
    });
    expect(http.calls.map(([method, url]) => [method, url])).toEqual([
      ["POST", "https://auth.openai.com/api/accounts/deviceauth/usercode"],
      ["POST", "https://auth.openai.com/api/accounts/deviceauth/token"],
      ["POST", "https://auth.openai.com/api/accounts/deviceauth/token"],
      ["POST", "https://auth.openai.com/api/accounts/deviceauth/token"],
      ["POST", "https://auth.openai.com/oauth/token"],
    ]);
    expect(http.calls[0]![2].headers).toEqual({ "User-Agent": USER_AGENT, "Content-Type": "application/json" });
    expect(http.calls.at(-1)![2].headers).toEqual({
      "User-Agent": USER_AGENT,
      "Content-Type": "application/x-www-form-urlencoded",
    });
    expect(http.calls[0]![2].timeoutMs).toBe(30_000);
    expect(inspect(credentials)).not.toContain("refresh-secret");
  });

  test("browser login validates state and uses PKCE", async () => {
    const http = new FakeHttp(new HttpResponse(200, tokens()));
    const urls: string[] = [];
    const auth = new OpenAIAuth({ request: http.request });
    const credentials = await auth.browserLogin((url) => void urls.push(url), {
      awaitCallback: callbackWith(urls, "code=browser-code"),
    });
    const url = new URL(urls[0]!);
    expect(url.origin + url.pathname).toBe("https://auth.openai.com/oauth/authorize");
    const query = url.searchParams;
    expect([...query.keys()]).toEqual([
      "response_type",
      "client_id",
      "redirect_uri",
      "scope",
      "code_challenge",
      "code_challenge_method",
      "id_token_add_organizations",
      "codex_cli_simplified_flow",
      "state",
      "originator",
    ]);
    const form = http.calls[0]![2].form!;
    const verifier = form.code_verifier!;
    expect(query.get("code_challenge")).toBe(createHash("sha256").update(verifier).digest("base64url"));
    expect(query.get("code_challenge_method")).toBe("S256");
    expect(query.get("redirect_uri")).toBe("http://localhost:1455/auth/callback");
    expect(query.get("scope")).toBe("openid profile email offline_access");
    expect(query.get("client_id")).toBe("app_EMoamEEZ73f0CkXaXp7hrann");
    expect(query.get("originator")).toBe("yian");
    expect(query.get("state")).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(verifier).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(form).toEqual({
      grant_type: "authorization_code",
      code: "browser-code",
      code_verifier: verifier,
      client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
      redirect_uri: "http://localhost:1455/auth/callback",
    });
    expect(credentials.accountId).toBe("account-test");
  });

  test("browser state mismatch never exchanges the code", async () => {
    const http = new FakeHttp();
    const error = await rejected(
      new OpenAIAuth({ request: http.request }).browserLogin(() => {}, {
        awaitCallback: (uri) => uri + "?code=secret-code&state=wrong",
      }),
    );
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toMatch(/state/);
    expect(http.calls).toEqual([]);
  });

  const REDIRECT = "http://localhost:1455/auth/callback";
  // Each row turns the valid callback URL (the redirect URI with the sent state) into the result.
  const invalidCallbacks: [string, (valid: string) => unknown, string][] = [
    ["state=bad", () => REDIRECT + "?state=bad&code=c", "OpenAI login state did not match"],
    [
      "state=bad&state=also-bad",
      () => REDIRECT + "?state=bad&state=also-bad&code=c",
      "OpenAI login state did not match",
    ],
    ["a doubled ?", (valid) => valid.replace("?", "??") + "&code=c", "OpenAI login state did not match"],
    ["error=denied", (valid) => valid + "&error=denied", "OpenAI browser authorization was denied"],
    [
      "code=one&code=two",
      (valid) => valid + "&code=one&code=two",
      "OpenAI callback did not include an authorization code",
    ],
    ["empty code", (valid) => valid + "&code=", "OpenAI callback did not include an authorization code"],
    ["no code", (valid) => valid + "&other=1", "OpenAI callback did not include an authorization code"],
    [
      "another host",
      (valid) => valid.replace("localhost", "127.0.0.1") + "&code=c",
      "OpenAI returned an invalid callback location",
    ],
    [
      "another scheme",
      (valid) => valid.replace("http:", "https:") + "&code=c",
      "OpenAI returned an invalid callback location",
    ],
    [
      "another path",
      (valid) => valid.replace("/auth/callback", "/auth/callbackx") + "&code=c",
      "OpenAI returned an invalid callback location",
    ],
    ["null", () => null, "OpenAI returned an invalid browser callback"],
    ["0", () => 0, "OpenAI returned an invalid browser callback"],
  ];
  for (const [label, result, message] of invalidCallbacks) {
    test(`browser callback rejects invalid or denied responses: ${label}`, async () => {
      const http = new FakeHttp();
      const urls: string[] = [];
      const awaitCallback = (uri: string) => result(uri + "?state=" + new URL(urls[0]!).searchParams.get("state"));
      const error = await rejected(
        new OpenAIAuth({ request: http.request }).browserLogin((url) => void urls.push(url), {
          awaitCallback: awaitCallback as (uri: string) => string,
        }),
      );
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(message);
      expect(http.calls).toEqual([]);
    });
  }

  test("refresh preserves missing metadata and rejects an account switch", async () => {
    const http = new FakeHttp(new HttpResponse(200, { access_token: "new", expires_in: 3600 }));
    const auth = new OpenAIAuth({ request: http.request, scheduler: virtualScheduler(1_000_000) });
    const updated = await auth.refresh(OLD());
    expect(updated).toBeInstanceOf(OAuthCredentials);
    const { accessToken, refreshToken, expiresAt, accountId, residency } = updated;
    const fields = [accessToken, refreshToken, expiresAt, accountId, residency];
    expect(fields).toEqual(["new", "old-refresh", 4600, "acct", "eu"]);
    expect(http.calls[0]![2].form).toEqual({
      grant_type: "refresh_token",
      refresh_token: "old-refresh",
      client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
    });
    expect(http.calls[0]![1]).toBe("https://auth.openai.com/oauth/token");
    const error = await rejected(
      new OpenAIAuth({ request: new FakeHttp(new HttpResponse(200, tokens("different"))).request }).refresh(OLD()),
    );
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toMatch(/account/);
  });

  const refusedRefreshes: [string, HttpResponse, string][] = [
    [
      "an OAuth error string",
      new HttpResponse(400, { error: "invalid_grant", error_description: "token old-refresh was revoked" }),
      "OpenAI authentication request failed (HTTP 400, invalid_grant); sign in again",
    ],
    [
      "an error record's code",
      new HttpResponse(401, { error: { code: "refresh_token_expired", message: "old-refresh expired" } }),
      "OpenAI authentication request failed (HTTP 401, refresh_token_expired); sign in again",
    ],
    [
      "a known code that leaves the grant usable",
      new HttpResponse(400, { error: "invalid_scope" }),
      "OpenAI authentication request failed (HTTP 400, invalid_scope)",
    ],
    [
      "a code off the known list",
      new HttpResponse(400, { error: "old-refresh" }),
      "OpenAI authentication request failed (HTTP 400)",
    ],
    ["no body", new HttpResponse(502, null), "OpenAI authentication request failed (HTTP 502)"],
  ];
  for (const [label, response, message] of refusedRefreshes) {
    test(`a refused refresh names the token endpoint's error code: ${label}`, async () => {
      const error = await rejected(new OpenAIAuth({ request: new FakeHttp(response).request }).refresh(OLD()));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(message);
      expect((error as AuthError).status).toBe(response.status);
      expect(error.cause).toBeUndefined();
    });
  }

  const invalidExpiries: [string, unknown][] = [
    ["10n ** 1000n", 10n ** 1000n],
    ["NaN", NaN],
    ["Infinity", Infinity],
    ["true", true],
    ['{"$float": "inf"}', { $float: "inf" }],
    ["a string", "3600"],
    ["null", null],
    ["0", 0n],
    ["-1", -1n],
  ];
  for (const [label, expiry] of invalidExpiries) {
    test(`an invalid expiry is rejected: ${label}`, async () => {
      const http = new FakeHttp(new HttpResponse(200, tokens("account-test", { expires_in: expiry })));
      const previous = new OAuthCredentials({
        accessToken: "old",
        refreshToken: "refresh",
        expiresAt: 1000,
        accountId: "account-test",
      });
      const error = await rejected(new OpenAIAuth({ request: http.request }).refresh(previous));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("OpenAI returned invalid credential expiry");
    });
  }

  describe("account metadata", () => {
    /** Refresh `OLD()` with `data`. */
    function refreshWith(data: Record<string, unknown>): Promise<OAuthCredentials> {
      return new OpenAIAuth({ request: new FakeHttp(new HttpResponse(200, data)).request }).refresh(OLD());
    }

    test("an id token and access token naming different accounts are refused", async () => {
      const error = await rejected(refreshWith(tokens("acct", { id_token: jwt("other") })));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("OpenAI refresh changed account identity");
    });

    test("a non-text account is refused", async () => {
      for (const token of [jwt(""), jwt("acct", { chatgpt_account_id: 5 })]) {
        const error = await rejected(refreshWith({ access_token: token, expires_in: 3600n }));
        expect(error).toBeInstanceOf(AuthError);
        expect(error.message).toBe("OpenAI returned invalid account metadata");
      }
    });

    test("a top-level account wins over the auth claim", async () => {
      const result = await refreshWith({ access_token: jwt("other", { chatgpt_account_id: "acct" }) });
      expect(result.accountId).toBe("acct");
    });

    test("organizations[0].id is used only when both chatgpt ids are absent", async () => {
      const fallback = await refreshWith({ access_token: jwt(null, { organizations: [{ id: "acct" }] }) });
      expect(fallback.accountId).toBe("acct");
      const ignored = await refreshWith({ access_token: jwt("acct", { organizations: [{ id: "org-other" }] }) });
      expect(ignored.accountId).toBe("acct");
      const other = jwt(null, { organizations: [{ id: "org-other" }] });
      const switched = await rejected(refreshWith({ access_token: other }));
      expect(switched.message).toBe("OpenAI refresh changed account identity");
    });

    test("residency is read from either claim and no_constraint clears it", async () => {
      const cleared = await refreshWith({ access_token: jwt("acct", { chatgpt_compute_residency: "no_constraint" }) });
      expect(cleared.residency).toBeNull();
      const auth = { chatgpt_account_id: "acct", chatgpt_compute_residency: "us" };
      const fromAuth = await refreshWith({ access_token: jwt(null, { "https://api.openai.com/auth": auth }) });
      expect(fromAuth.residency).toBe("us");
      const kept = await refreshWith({ access_token: jwt("acct") });
      expect(kept.residency).toBe("eu");
    });

    test("tokens that aren't JWTs keep the previous account on refresh and fail a login", async () => {
      const kept = await refreshWith({ access_token: "opaque.token", refresh_token: "r2" });
      expect(kept.accountId).toBe("acct");
      expect(kept.refreshToken).toBe("r2");
      const urls: string[] = [];
      const http = new FakeHttp(new HttpResponse(200, { access_token: "opaque", refresh_token: "r" }));
      const error = await rejected(
        new OpenAIAuth({ request: http.request }).browserLogin((url) => void urls.push(url), {
          awaitCallback: callbackWith(urls, "code=c"),
        }),
      );
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Invalid subscription credentials");
    });
  });

  test("API key validation lists models through the official endpoint", async () => {
    const data = [{ id: "z-model" }, { id: "a-model" }, { id: "z-model" }];
    const http = new FakeHttp(new HttpResponse(200, { data }));
    const models = await new OpenAIAuth({ request: http.request }).apiModels(new ApiKeyCredentials("secret"));
    expect(models).toEqual(["a-model", "z-model"]);
    expect(Object.isFrozen(models)).toBe(true);
    expect(http.calls[0]!.slice(0, 2)).toEqual(["GET", "https://api.openai.com/v1/models"]);
    expect(http.calls[0]![2].headers.Authorization).toBe("Bearer secret");
  });

  test("API key validation sends organization, project and headers", async () => {
    const http = new FakeHttp(new HttpResponse(200, { data: [{ id: "m" }] }));
    const credentials = new ApiKeyCredentials("secret", { "X-Gateway": GATEWAY });
    const auth = new OpenAIAuth({ request: http.request });
    expect(await auth.apiModels(credentials, { organization: ORG, project: PROJECT })).toEqual(["m"]);
    expect(http.calls[0]![2].headers).toEqual({
      "User-Agent": USER_AGENT,
      "X-Gateway": GATEWAY,
      "OpenAI-Organization": ORG,
      "OpenAI-Project": PROJECT,
      Authorization: "Bearer secret",
    });
  });

  test("an organization mismatch is an HTTP error without values", async () => {
    const body = { error: { code: "mismatched_organization", message: `${ORG} ${PROJECT}` } };
    const http = new FakeHttp(new HttpResponse(401, body));
    const credentials = new ApiKeyCredentials("secret", { "X-Gateway": GATEWAY });
    const error = await rejected(
      new OpenAIAuth({ request: http.request }).apiModels(credentials, { organization: ORG, project: PROJECT }),
    );
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("OpenAI authentication request failed (HTTP 401)");
    expect((error as AuthError).status).toBe(401);
    assertNoSettings(error);
    expect(http.calls).toHaveLength(1);
  });

  const invalidSettings: [string, ApiKeyCredentials, { organization?: unknown; project?: unknown }][] = [
    ["organization with a space", new ApiKeyCredentials("secret"), { organization: ORG + " x" }],
    ["empty project", new ApiKeyCredentials("secret"), { project: "" }],
    ["number organization", new ApiKeyCredentials("secret"), { organization: 5 }],
    // OpenAI takes these through organization / project, not as headers.
    ["OpenAI-Organization header", new ApiKeyCredentials("secret", { "OpenAI-Organization": ORG }), {}],
    ["openai-project header", new ApiKeyCredentials("secret", { "openai-project": PROJECT }), {}],
  ];
  for (const [label, credentials, settings] of invalidSettings) {
    test(`invalid connection settings are refused before any request: ${label}`, async () => {
      const http = new FakeHttp();
      const error = await rejected(new OpenAIAuth({ request: http.request }).apiModels(credentials, settings));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toStartWith("Invalid OpenAI connection settings");
      expect(error.cause).toBeUndefined();
      assertNoSettings(error);
      expect(http.calls).toEqual([]);
    });
  }

  test("a transport error never echoes connection settings or chains its cause", async () => {
    const broken: AuthRequest = () => {
      throw new Error(`${GATEWAY} ${ORG} ${PROJECT} sensitive-token-in-url`);
    };
    const credentials = new ApiKeyCredentials("secret", { "X-Gateway": GATEWAY });
    const auth = new OpenAIAuth({ request: broken });
    const error = await rejected(auth.apiModels(credentials, { organization: ORG, project: PROJECT }));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("OpenAI authentication request failed");
    expect(error.cause).toBeUndefined();
    assertNoSettings(error);
    expect(error.stack).not.toContain("sensitive");
    const asyncError = await rejected(
      new OpenAIAuth({ request: async () => Promise.reject(new Error("sensitive-token-in-url")) }).apiModels(
        new ApiKeyCredentials("key"),
      ),
    );
    expect(asyncError.message).toBe("OpenAI authentication request failed");
    expect(asyncError.cause).toBeUndefined();
  });

  const transportKinds: [ProviderError, string][] = [
    [new ProviderError("timeout", "HTTP request failed (timeout): sensitive"), "timed out"],
    [new ProviderError("connection", "HTTP request failed (connection): sensitive"), "could not connect"],
    [new ProviderError("protocol", "HTTP response is larger than 10 bytes: sensitive"), "reply too large"],
  ];
  for (const [transport, words] of transportKinds) {
    test(`a transport failure keeps its kind but not its text: ${transport.kind}`, async () => {
      const auth = new OpenAIAuth({ request: async () => Promise.reject(transport) });
      const error = await rejected(auth.apiModels(new ApiKeyCredentials("key")));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(`OpenAI authentication request failed (${words})`);
      expect(error.cause).toBeUndefined();
      expect(error.stack).not.toContain("sensitive");
    });
  }

  test("the subscription catalog uses account routing and visibility order", async () => {
    const http = new FakeHttp(
      new HttpResponse(200, {
        models: [
          { slug: "third", visibility: "list", priority: 3n },
          { slug: "hidden", visibility: "hide", priority: 0n },
          { slug: "first", visibility: "list", priority: 1n },
          { slug: "unavailable", visibility: "none", priority: -1n },
          { slug: "first", visibility: "list", priority: 2n },
        ],
      }),
    );
    const credentials = new OAuthCredentials({
      accessToken: "access",
      refreshToken: "refresh",
      expiresAt: 9000,
      accountId: "acct",
      residency: "eu",
    });
    const models = await new OpenAIAuth({ request: http.request }).subscriptionModels(credentials);
    expect(models).toEqual(["first", "third"]);
    expect(Object.isFrozen(models)).toBe(true);
    expect(http.calls[0]!.slice(0, 2)).toEqual([
      "GET",
      "https://chatgpt.com/backend-api/codex/models?client_version=0.155.1",
    ]);
    expect(http.calls[0]![2].headers).toEqual({
      "User-Agent": USER_AGENT,
      Authorization: "Bearer access",
      "ChatGPT-Account-Id": "acct",
      originator: "yian",
      "x-openai-internal-codex-residency": "eu",
    });
    for (const clientVersion of ["", "1 2", null, 5]) {
      const error = await rejected(
        new OpenAIAuth({ request: new FakeHttp().request }).subscriptionModels(credentials, { clientVersion }),
      );
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe("Invalid model catalog client version");
    }
  });

  const malformedCatalogs: [string, "api" | "subscription", unknown, string][] = [
    ["empty api list", "api", { data: [] }, "OpenAI returned an invalid or empty model list"],
    ["empty api id", "api", { data: [{ id: "" }] }, "OpenAI returned an invalid or empty model list"],
    ["empty models", "subscription", { models: [] }, "OpenAI returned no available subscription models"],
    [
      "only hidden models",
      "subscription",
      { models: [{ slug: "hidden", visibility: "hide", priority: 0n }] },
      "OpenAI returned no available subscription models",
    ],
    [
      "boolean priority",
      "subscription",
      { models: [{ slug: "bad", visibility: "list", priority: true }] },
      "OpenAI returned an invalid subscription model list",
    ],
    [
      "float priority",
      "subscription",
      { models: [{ slug: "bad", visibility: "list", priority: 1.5 }] },
      "OpenAI returned an invalid subscription model list",
    ],
    [
      "unknown visibility",
      "subscription",
      { models: [{ slug: "bad", visibility: "public", priority: 1n }] },
      "OpenAI returned an invalid subscription model list",
    ],
  ];
  for (const [label, kind, payload, message] of malformedCatalogs) {
    test(`a model catalog rejects malformed or empty lists: ${label}`, async () => {
      const auth = new OpenAIAuth({ request: new FakeHttp(new HttpResponse(200, payload)).request });
      const error = await rejected(
        kind === "api" ? auth.apiModels(new ApiKeyCredentials("key")) : auth.subscriptionModels(SUBSCRIPTION()),
      );
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(message);
    });
  }

  const methods: Record<string, (auth: OpenAIAuth) => Promise<unknown>> = {
    api: (auth) => auth.apiModels(new ApiKeyCredentials("key")),
    subscription: (auth) => auth.subscriptionModels(SUBSCRIPTION()),
    refresh: (auth) => auth.refresh(SUBSCRIPTION()),
    device: (auth) => auth.deviceLogin(() => {}),
  };
  for (const [method, call] of Object.entries(methods)) {
    test(`HTTP failures do not echo secrets or fall back: ${method}`, async () => {
      const http = new FakeHttp(new HttpResponse(401, { error: "secret-do-not-print" }));
      const error = await rejected(call(new OpenAIAuth({ request: http.request })));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toMatch(/HTTP 401/);
      expect(error.message).not.toContain("secret");
      expect(http.calls).toHaveLength(1);
    });
  }

  test("an HTTP failure sets the status on AuthError", async () => {
    for (const status of [401, 403, 404]) {
      const http = new FakeHttp(new HttpResponse(status, { error: "secret-do-not-print" }));
      const error = await rejected(new OpenAIAuth({ request: http.request }).subscriptionModels(SUBSCRIPTION()));
      expect(error).toBeInstanceOf(AuthError);
      expect((error as AuthError).status).toBe(status);
    }
    const http = new FakeHttp(new HttpResponse(403, { error: "secret-do-not-print" }));
    const error = await rejected(new OpenAIAuth({ request: http.request }).refresh(SUBSCRIPTION()));
    expect(error).toBeInstanceOf(AuthError);
    expect((error as AuthError).status).toBe(403);
  });

  test("an AuthError without an HTTP status has a null status and no cause", () => {
    const error = new AuthError("x");
    expect(error.status).toBeNull();
    expect("cause" in error).toBe(false);
    const cause = new Error("inner");
    const wrapped = new AuthError("x", { cause });
    expect(wrapped.cause).toBe(cause);
    expect(wrapped.status).toBeNull();
    expect(new AuthError("x", { status: 401 }).status).toBe(401);
  });

  test("cancellation comes before the network", async () => {
    const controller = new AbortController();
    const reason = new Error("stopped by host");
    controller.abort(reason);
    const http = new FakeHttp();
    const auth = new OpenAIAuth({ request: http.request, signal: controller.signal });
    expect(await rejected(auth.deviceLogin(() => {}))).toBe(reason);
    expect(await rejected(auth.browserLogin(() => {}, { awaitCallback: (uri) => uri }))).toBe(reason);
    expect(await rejected(auth.refresh(SUBSCRIPTION()))).toBe(reason);
    expect(http.calls).toEqual([]);
  });

  for (const cancel of [false, true]) {
    test(`device polling has a bounded cancellable wait: ${cancel ? "cancel" : "timeout"}`, async () => {
      const scheduler = virtualScheduler(1_000_000);
      const http = new FakeHttp(
        new HttpResponse(200, { device_auth_id: "id", usercode: "code", interval: "5" }),
        new HttpResponse(404, {}),
      );
      const controller = new AbortController();
      const reason = new Error("stopped by host");
      if (cancel) void scheduler.sleep(500).then(() => controller.abort(reason));
      const auth = new OpenAIAuth({ request: http.request, scheduler, signal: controller.signal });
      const error = await rejected(auth.deviceLogin(() => {}, { timeoutMs: 1000 }));
      if (cancel) {
        expect(error).toBe(reason);
      } else {
        expect(error).toBeInstanceOf(AuthError);
        expect(error.message).toBe("OpenAI login timed out");
      }
      expect(http.calls).toHaveLength(2);
      expect(scheduler.now()).toBeLessThanOrEqual(1_001_000);
    });
  }

  const badIntervals: [string, unknown][] = [
    ['"nan"', "nan"],
    ['"inf"', "inf"],
    ['"infinity"', "infinity"],
    ["true", true],
    ['"1__0"', "1__0"],
    ['"_1"', "_1"],
    ['"1_"', "1_"],
    ['""', ""],
    ['"0x10"', "0x10"],
    ["[]", []],
    ['{"$float": "inf"}', { $float: "inf" }],
    ["null", null],
    ["10n ** 400n", 10n ** 400n],
  ];
  for (const [label, interval] of badIntervals) {
    test(`device login rejects a non-finite or non-numeric interval: ${label}`, async () => {
      const http = new FakeHttp(new HttpResponse(200, { device_auth_id: "id", user_code: "code", interval }));
      const display: string[] = [];
      const error = await rejected(
        new OpenAIAuth({ request: http.request }).deviceLogin((_url, code) => void display.push(code)),
      );
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("OpenAI returned an invalid device polling interval");
      expect(http.calls).toHaveLength(1);
      expect(display).toEqual([]);
    });
  }

  for (const [interval, gapMs] of [
    ["0.5", 1000],
    ["1_0", 10_000],
    [" 2.5e0 ", 2500],
    [3n, 3000],
    [1.5, 1500],
  ] as const) {
    test(`device polling interval ${JSON.stringify(String(interval))} waits ${gapMs} ms`, async () => {
      const scheduler = virtualScheduler(1_000_000);
      const http = new FakeHttp(
        new HttpResponse(200, { device_auth_id: "id", user_code: "code", interval }),
        new HttpResponse(403, {}),
        new HttpResponse(200, { authorization_code: "c", code_verifier: "v" }),
        new HttpResponse(200, tokens()),
      );
      http.clock = scheduler;
      await new OpenAIAuth({ request: http.request, scheduler }).deviceLogin(() => {});
      expect(http.times[2]! - http.times[1]!).toBe(gapMs);
    });
  }

  test("a huge device interval doesn't overflow the sleep; the login times out", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const http = new FakeHttp(
      new HttpResponse(200, { device_auth_id: "id", user_code: "code", interval: 1e306 }),
      new HttpResponse(404, {}),
    );
    const error = await rejected(
      new OpenAIAuth({ request: http.request, scheduler }).deviceLogin(() => {}, { timeoutMs: 5000 }),
    );
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("OpenAI login timed out");
    expect(scheduler.now()).toBe(1_005_000);
  });

  const invalidDeviceData: [string, Record<string, unknown>][] = [
    ["missing device_auth_id", { user_code: "code" }],
    ["user_code with a space", { device_auth_id: "id", user_code: "AB CD" }],
    ["user_code null beside usercode", { device_auth_id: "id", user_code: null, usercode: "X" }],
    ["number device_auth_id", { device_auth_id: 5n, user_code: "code" }],
  ];
  for (const [label, data] of invalidDeviceData) {
    test(`invalid device authorization data is refused: ${label}`, async () => {
      const http = new FakeHttp(new HttpResponse(200, data));
      const error = await rejected(new OpenAIAuth({ request: http.request }).deviceLogin(() => {}));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("OpenAI returned invalid device authorization data");
      expect(http.calls).toHaveLength(1);
    });
  }

  test("usercode is accepted when user_code is absent", async () => {
    const http = new FakeHttp(
      new HttpResponse(200, { device_auth_id: "id", usercode: "X-Y" }),
      new HttpResponse(200, { authorization_code: "c", code_verifier: "v" }),
      new HttpResponse(200, tokens()),
    );
    const display: string[] = [];
    await new OpenAIAuth({ request: http.request }).deviceLogin((_url, code) => void display.push(code));
    expect(display).toEqual(["X-Y"]);
    expect(http.calls[1]![2].jsonBody).toEqual({ device_auth_id: "id", user_code: "X-Y" });
  });

  test("a failed poll and a missing authorization code are refused", async () => {
    const failed = new FakeHttp(
      new HttpResponse(200, { device_auth_id: "id", user_code: "code" }),
      new HttpResponse(500, { error: "boom" }),
    );
    const error = await rejected(new OpenAIAuth({ request: failed.request }).deviceLogin(() => {}));
    expect(error.message).toBe("OpenAI authentication request failed (HTTP 500)");
    expect((error as AuthError).status).toBe(500);
    const missing = new FakeHttp(
      new HttpResponse(200, { device_auth_id: "id", user_code: "code" }),
      new HttpResponse(200, { code_verifier: "v" }),
    );
    const missingError = await rejected(new OpenAIAuth({ request: missing.request }).deviceLogin(() => {}));
    expect(missingError).toBeInstanceOf(AuthError);
    expect(missingError.message).toBe("OpenAI returned invalid authorization data");
    expect((missingError as AuthError).status).toBeNull();
    expect(missing.calls).toHaveLength(2);
  });

  test("the request timeout and the login timeout are checked", async () => {
    for (const requestTimeoutMs of [0, -1, NaN, Infinity]) {
      const error = thrown(() => new OpenAIAuth({ requestTimeoutMs }));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe("Invalid authentication request timeout");
    }
    const http = new FakeHttp();
    const auth = new OpenAIAuth({ request: http.request, requestTimeoutMs: 1 });
    for (const timeoutMs of [0, NaN, Infinity]) {
      for (const login of [
        () => auth.deviceLogin(() => {}, { timeoutMs }),
        () => auth.browserLogin(() => {}, { awaitCallback: (uri) => uri, timeoutMs }),
      ]) {
        const error = await rejected(login());
        expect(error).toBeInstanceOf(RangeError);
        expect(error.message).toBe("Invalid authentication timeout");
      }
    }
    expect(http.calls).toEqual([]);
  });

  class SubResponse extends HttpResponse {}
  const invalidResponses: [string, unknown, string][] = [
    ["a plain object", { status: 200, data: { data: [{ id: "m" }] } }, "returned an invalid response"],
    ["a subclass", new SubResponse(200, { data: [{ id: "m" }] }), "returned an invalid response"],
    ["undefined", undefined, "returned an invalid response"],
    ["status 99", new HttpResponse(99, {}), "returned an invalid response"],
    ["status 600", new HttpResponse(600, {}), "returned an invalid response"],
    ["status 1.5", new HttpResponse(1.5, {}), "returned an invalid response"],
    ["an array body", new HttpResponse(200, [{ id: "m" }]), "returned malformed data"],
    ["a null body", new HttpResponse(200, null), "returned malformed data"],
  ];
  for (const [label, response, message] of invalidResponses) {
    test(`a response that isn't an HttpResponse with a valid status is refused: ${label}`, async () => {
      const auth = new OpenAIAuth({ request: async () => response as HttpResponse });
      const error = await rejected(auth.apiModels(new ApiKeyCredentials("key")));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(`OpenAI authentication ${message}`);
    });
  }

  test("the deadline aborts an in-flight request", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const signals: (AbortSignal | undefined)[] = [];
    const auth = new OpenAIAuth({
      request: (_method, _url, options) => {
        signals.push(options.signal);
        return never();
      },
      scheduler,
    });
    const urls: string[] = [];
    const error = await rejected(
      auth.browserLogin((url) => void urls.push(url), { awaitCallback: callbackWith(urls, "code=c"), timeoutMs: 1000 }),
    );
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("OpenAI browser login timed out");
    expect(signals).toHaveLength(1);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[0]!.reason).toBe(error);
    expect(scheduler.now()).toBe(1_001_000);
  });

  test("host callbacks are bounded by the deadline", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const http = new FakeHttp(new HttpResponse(200, { device_auth_id: "id", user_code: "code" }));
    const auth = new OpenAIAuth({ request: http.request, scheduler });
    for (const login of [
      () => auth.browserLogin(() => {}, { awaitCallback: never, timeoutMs: 1000 }),
      () => auth.browserLogin(never, { awaitCallback: (uri) => uri, timeoutMs: 1000 }),
    ]) {
      const error = await rejected(login());
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("OpenAI browser login timed out");
    }
    const error = await rejected(auth.deviceLogin(never, { timeoutMs: 1000 }));
    expect(error).toBeInstanceOf(AuthError);
    expect(error.message).toBe("OpenAI login timed out");
    expect(http.calls).toHaveLength(1);
  });

  test("a stop while openUrl hangs rejects with the reason", async () => {
    const scheduler = virtualScheduler(1_000_000);
    const controller = new AbortController();
    const reason = new Error("stopped by host");
    void scheduler.sleep(100).then(() => controller.abort(reason));
    const auth = new OpenAIAuth({ request: new FakeHttp().request, scheduler, signal: controller.signal });
    const error = await rejected(auth.browserLogin(never, { awaitCallback: (uri) => uri }));
    expect(error).toBe(reason);
    expect(scheduler.now()).toBe(1_000_100);
  });

  for (const phase of ["headers", "body", "idle_headers", "idle_body"] as const) {
    for (const cancel of [false, true]) {
      const label = `${phase}, ${cancel ? "cancel" : "timeout"}`;
      test(`http trickle respects the deadline, cancellation, and closes the socket: ${label}`, async () => {
        const controller = new AbortController();
        const reason = new Error("stopped by host");
        let peerClosed!: () => void;
        const closed = new Promise<void>((resolve) => (peerClosed = resolve));
        const serverErrors: unknown[] = [];
        const timers: ReturnType<typeof setInterval>[] = [];
        const sockets: net.Socket[] = [];
        const wait = new AbortController();
        const server = net.createServer((socket) => {
          sockets.push(socket);
          socket.on("error", () => {});
          socket.on("close", () => peerClosed());
          let received = "";
          socket.on("data", (chunk) => {
            if (received.includes("\r\n\r\n")) return;
            received += chunk.toString("latin1");
            if (!received.includes("\r\n\r\n")) return;
            const head = phase.endsWith("headers")
              ? "HTTP/1.1 200 OK\r\nX-Slow: "
              : "HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\n";
            socket.write(head, () => {
              if (cancel) controller.abort(reason);
            });
            if (phase.startsWith("idle")) return;
            let sent = 0;
            const timer = setInterval(() => {
              if (sent++ >= 20 || socket.destroyed) return clearInterval(timer);
              socket.write(" ");
            }, 50);
            timers.push(timer);
          });
        });
        server.on("error", (error) => serverErrors.push(error));
        await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
        const { port } = server.address() as AddressInfo;
        try {
          const auth = new OpenAIAuth({
            request: (method, _url, options) => request(method, `http://127.0.0.1:${port}/`, options),
            requestTimeoutMs: 250,
            signal: controller.signal,
            scheduler: realScheduler,
          });
          const start = performance.now();
          const error = await rejected(auth.apiModels(new ApiKeyCredentials("test-only-key")));
          if (cancel) {
            expect(error).toBe(reason);
          } else {
            expect(error).toBeInstanceOf(AuthError);
            expect(error.message).toBe("OpenAI authentication request failed (timed out)");
          }
          expect(performance.now() - start).toBeLessThan(700);
          const peer = await Promise.race([
            closed.then(() => true),
            realScheduler.sleep(1000, wait.signal).then(
              () => false,
              () => false,
            ),
          ]);
          expect(peer).toBe(true);
        } finally {
          wait.abort();
          for (const timer of timers) clearInterval(timer);
          // A socket still open would hold server.close until the runner's timeout.
          for (const socket of sockets) socket.destroy();
          await new Promise((resolve) => server.close(resolve));
        }
        expect(serverErrors).toEqual([]);
      });
    }
  }
});

/** Send `data` to the login callback listener and return everything it answers before closing. */
function callbackExchange(data: string | Buffer, options: { end?: boolean } = {}): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const socket = net.connect({ host: "127.0.0.1", port: 1455 }, () => {
      if (options.end) socket.end(data);
      else socket.write(data);
    });
    const chunks: Buffer[] = [];
    socket.on("data", (chunk) => chunks.push(chunk));
    socket.on("error", reject);
    socket.on("close", () => resolve(Buffer.concat(chunks)));
  });
}

/** Open a connection to the login callback listener that sends nothing. */
function idleConnection(): net.Socket {
  const socket = net.connect({ host: "127.0.0.1", port: 1455 });
  socket.on("error", () => {});
  return socket;
}

/** A valid callback request carrying the state of the authorize `url`. */
function callbackRequest(url: string, version = "HTTP/1.1"): string {
  const state = new URL(url).searchParams.get("state");
  return `GET /auth/callback?code=code&state=${state} ${version}\r\nHost: localhost:1455\r\n\r\n`;
}

/** The listener's exact answer for `status` and `body`. */
function callbackAnswer(status: string, body: string): string {
  return (
    `HTTP/1.1 ${status}\r\nContent-Type: text/plain; charset=utf-8\r\n` +
    `Content-Length: ${Buffer.byteLength(body)}\r\nConnection: close\r\n\r\n${body}`
  );
}

/** Bind 127.0.0.1:1455 and release it, proving no listener still holds it. */
async function expectCallbackPortFree(): Promise<void> {
  const server = net.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen({ host: "127.0.0.1", port: 1455 }, resolve);
  });
  await new Promise((resolve) => server.close(resolve));
}

describe("browser callback listener", () => {
  test("the listener is closed after success", async () => {
    const http = new FakeHttp(new HttpResponse(200, tokens()));
    const replies: Promise<Buffer>[] = [];
    const auth = new OpenAIAuth({ request: http.request, scheduler: realScheduler });
    const credentials = await auth.browserLogin((url) => void replies.push(callbackExchange(callbackRequest(url))), {
      timeoutMs: 3000,
    });
    const reply = (await replies[0]!).toString("latin1");
    expect(reply).toStartWith("HTTP/1.1 200 OK");
    expect(reply).toEndWith("Authorization received. Return to Yian.");
    expect(credentials.accountId).toBe("account-test");
    expect(http.calls[0]![2].form!.code).toBe("code");
    await expectCallbackPortFree();
  });

  test("the listener is closed when the user cancels", async () => {
    const unhandled: unknown[] = [];
    const guard = (reason: unknown) => void unhandled.push(reason);
    process.on("unhandledRejection", guard);
    try {
      const cancelled = new Error("cancelled by user");
      const auth = new OpenAIAuth({ request: new FakeHttp().request, scheduler: realScheduler });
      const error = await rejected(
        auth.browserLogin(() => {
          throw cancelled;
        }),
      );
      expect(error).toBe(cancelled);
      await realScheduler.sleep(50);
    } finally {
      process.off("unhandledRejection", guard);
    }
    expect(unhandled).toEqual([]);
    await expectCallbackPortFree();
  });

  test("the callback deadline survives an idle local connection", async () => {
    const sockets: net.Socket[] = [];
    const auth = new OpenAIAuth({ request: new FakeHttp().request, scheduler: realScheduler });
    const start = performance.now();
    try {
      const error = await rejected(auth.browserLogin(() => void sockets.push(idleConnection()), { timeoutMs: 300 }));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toMatch(/timed out/);
      expect(performance.now() - start).toBeLessThan(800);
    } finally {
      for (const socket of sockets) socket.destroy();
    }
    await expectCallbackPortFree();
  });

  test("an idle connection opened first doesn't hold up a real callback", async () => {
    const http = new FakeHttp(new HttpResponse(200, tokens()));
    const sockets: net.Socket[] = [];
    const replies: Promise<Buffer>[] = [];
    const auth = new OpenAIAuth({ request: http.request, scheduler: realScheduler });
    const start = performance.now();
    try {
      const credentials = await auth.browserLogin(
        (url) => {
          sockets.push(idleConnection());
          replies.push(callbackExchange(callbackRequest(url)));
        },
        { timeoutMs: 3000 },
      );
      expect(credentials.accountId).toBe("account-test");
      expect(performance.now() - start).toBeLessThan(1000);
      expect((await replies[0]!).toString("latin1")).toStartWith("HTTP/1.1 200 OK");
    } finally {
      for (const socket of sockets) socket.destroy();
    }
    await expectCallbackPortFree();
  });

  for (const cancel of [false, true]) {
    const label = cancel ? "cancel" : "timeout";
    test(`the callback checks the deadline and cancellation during slow headers: ${label}`, async () => {
      const controller = new AbortController();
      const reason = new Error("stopped by host");
      const sockets: net.Socket[] = [];
      const timers: ReturnType<typeof setInterval>[] = [];
      const trickle = () => {
        const socket = idleConnection();
        sockets.push(socket);
        let sent = 0;
        const timer = setInterval(() => {
          if (sent++ >= 20 || socket.destroyed) return clearInterval(timer);
          socket.write("G", () => {
            if (cancel) controller.abort(reason);
          });
        }, 50);
        timers.push(timer);
      };
      const auth = new OpenAIAuth({
        request: new FakeHttp().request,
        scheduler: realScheduler,
        signal: controller.signal,
      });
      const start = performance.now();
      try {
        const error = await rejected(auth.browserLogin(trickle, { timeoutMs: 250 }));
        if (cancel) expect(error).toBe(reason);
        else expect(error).toBeInstanceOf(AuthError);
        expect(performance.now() - start).toBeLessThan(700);
      } finally {
        for (const timer of timers) clearInterval(timer);
        for (const socket of sockets) socket.destroy();
      }
      await expectCallbackPortFree();
    });
  }

  test("invalid requests get 400 or 404 and a valid HTTP/1.0 callback still succeeds", async () => {
    const http = new FakeHttp(new HttpResponse(200, tokens()));
    const bad = callbackAnswer("400 Bad Request", "Invalid callback request.");
    const unknown = callbackAnswer("404 Not Found", "Unknown callback path.");
    const replies: [string, string][] = [];
    const auth = new OpenAIAuth({ request: http.request, scheduler: realScheduler });
    const credentials = await auth.browserLogin(
      async (url) => {
        const cases: [string | Buffer, string, boolean?][] = [
          ["POST /auth/callback HTTP/1.1\r\nHost: localhost:1455\r\n\r\n", bad],
          ["GET /other HTTP/1.1\r\n\r\n", unknown],
          ["GET //evil/auth/callback HTTP/1.1\r\n\r\n", unknown],
          ["GET /auth/callback HTTP/2.0\r\n\r\n", bad],
          ["GET /auth/callback HTTP/1.1 extra\r\n\r\n", bad],
          [Buffer.from("GET /auth/callback?x=\xe9 HTTP/1.1\r\n\r\n", "latin1"), bad],
          ["GET /auth/callback HTTP/1.1\r\nHost: localhost:1455\r\n", bad, true],
          [callbackRequest(url, "HTTP/1.0"), callbackAnswer("200 OK", "Authorization received. Return to Yian.")],
        ];
        for (const [data, expected, end] of cases) {
          replies.push([(await callbackExchange(data, { end })).toString("latin1"), expected]);
        }
      },
      { timeoutMs: 3000 },
    );
    for (const [reply, expected] of replies) expect(reply).toBe(expected);
    expect(replies).toHaveLength(8);
    expect(credentials.accountId).toBe("account-test");
    expect(http.calls).toHaveLength(1);
    await expectCallbackPortFree();
  });

  test("a callback with the wrong or no state gets 400 and the login waits for the real one", async () => {
    const http = new FakeHttp(new HttpResponse(200, tokens()));
    const mismatch = callbackAnswer("400 Bad Request", "Callback state did not match this login.");
    const replies: [string, string][] = [];
    const auth = new OpenAIAuth({ request: http.request, scheduler: realScheduler });
    const credentials = await auth.browserLogin(
      async (url) => {
        const state = new URL(url).searchParams.get("state");
        const cases: [string, string][] = [
          ["GET /auth/callback?code=old&state=stale HTTP/1.1\r\n\r\n", mismatch],
          ["GET /auth/callback?code=old HTTP/1.1\r\n\r\n", mismatch],
          [`GET /auth/callback?code=old&state=${state}&state=${state} HTTP/1.1\r\n\r\n`, mismatch],
          [callbackRequest(url), callbackAnswer("200 OK", "Authorization received. Return to Yian.")],
        ];
        for (const [data, expected] of cases) {
          replies.push([(await callbackExchange(data)).toString("latin1"), expected]);
        }
      },
      { timeoutMs: 3000 },
    );
    for (const [reply, expected] of replies) expect(reply).toBe(expected);
    expect(replies).toHaveLength(4);
    expect(credentials.accountId).toBe("account-test");
    expect(http.calls).toHaveLength(1);
    expect(http.calls[0]![2].form!.code).toBe("code");
    await expectCallbackPortFree();
  });

  test("a taken port is refused before the browser opens", async () => {
    const blocker = net.createServer();
    await new Promise<void>((resolve, reject) => {
      blocker.once("error", reject);
      blocker.listen({ host: "127.0.0.1", port: 1455 }, resolve);
    });
    const opened: string[] = [];
    try {
      const auth = new OpenAIAuth({ request: new FakeHttp().request, scheduler: realScheduler });
      const error = await rejected(auth.browserLogin((url) => void opened.push(url), { timeoutMs: 1000 }));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe("Could not open the local login callback; use device login");
      expect(error.cause).toBeUndefined();
      expect(opened).toEqual([]);
    } finally {
      await new Promise((resolve) => blocker.close(resolve));
    }
    await expectCallbackPortFree();
  });
});

const BUILT = join(import.meta.dirname, "..", "..", "dist", "tests", "llm_clients", "auth_both_runtimes.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/platform.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!fs.existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

describe("the both-runtime scenarios", () => {
  // More than the cross-process scenario's own 20 s limit, so that limit reports first.
  for (const [name, run] of Object.entries(SCENARIOS)) test(name, run, 30_000);

  test.skipIf(!NODE)(
    "all pass on Node, built",
    () => {
      const result = spawnSync(NODE!, [BUILT], { encoding: "utf8", timeout: 60_000 });
      const context = `signal: ${result.signal}\nstderr: ${result.stderr}\nstdout: ${result.stdout}`;
      expect(result.stdout.trim(), context).not.toBe("");
      const lines = result.stdout.trim().split("\n").map((line) => {
        let parsed: { name?: unknown; ok?: unknown; error?: string };
        try {
          parsed = JSON.parse(line);
        } catch {
          throw new Error(`not a JSON line: ${line}\n${context}`);
        }
        return parsed;
      });
      expect(lines.map((line) => line.name)).toEqual(Object.keys(SCENARIOS));
      for (const line of lines) expect(line, line.error).toEqual({ name: line.name, ok: true });
      expect(result.status, context).toBe(0);
    },
    90_000,
  );
});
