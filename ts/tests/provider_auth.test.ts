/**
 * Mirrors `tests/test_provider_auth.py`: the credential values, the store and `isAuthFailure`.
 * The OAuth flow, refresh HTTP and model-catalog tests go with `auth/openai.ts` and
 * `auth/http.ts`, as does `test_refresh_is_serialized_across_processes`.
 */

import { type ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { inspect } from "node:util";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import { ProviderError } from "../src/llm/errors.ts";
import {
  ApiKeyCredentials,
  AuthError,
  isAuthFailure,
  OAuthCredentials,
  ProviderSelection,
} from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { lockFile, realScheduler, type Scheduler, virtualScheduler } from "../src/platform.ts";
import { parseJson } from "../src/typesys/json.ts";

const PLATFORM_URL = pathToFileURL(join(import.meta.dir, "..", "src", "platform.ts")).href;

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
    ["not an object", "401", false],
    ["null", null, false],
  ];
  for (const [label, error, expected] of rows) {
    test(label, () => {
      expect(isAuthFailure(error)).toBe(expected);
    });
  }
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

  test("a store written by save reads back in Python's format", async () => {
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

  for (const cancel of [false, true]) {
    test(`the lock wait is bounded and cancellable: ${cancel ? "cancel" : "timeout"}`, async () => {
      await new AuthStore(file).save("openai", new ApiKeyCredentials("key"), "model");
      const scheduler = virtualScheduler(1_000_000);
      const controller = new AbortController();
      const reason = new Error("stopped");
      if (cancel) void scheduler.sleep(200).then(() => controller.abort(reason));
      const holder = lockFile(`${file}.lock`);
      try {
        const store = new AuthStore(file, { scheduler, signal: controller.signal, lockTimeoutMs: 500 });
        const error = await rejected(store.save("openai", new ApiKeyCredentials("new"), "model-new"));
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
