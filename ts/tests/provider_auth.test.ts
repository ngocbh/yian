/**
 * Mirrors `tests/test_provider_auth.py`: the credential values, the store and `isAuthFailure`.
 * The OAuth flow, refresh HTTP and model-catalog tests go with `auth/openai.ts` and
 * `auth/http.ts`, as does `test_refresh_is_serialized_across_processes`.
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
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
    ["undefined pair value", [["X-A", undefined]]],
    ["undefined Map value", new Map([["X-A", undefined]])],
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

  const refusals: [Record<string, unknown>, string][] = [
    [{ provider: "" }, "Invalid provider selection"],
    [{ method: "other" }, "Invalid provider selection"],
    [{ endpoint: "has space" }, "Invalid provider selection"],
    [{ headerNames: "X-A" }, "Invalid provider selection"],
    [
      { method: "subscription", organization: "org-1" },
      "organization and project are only for an openai API-key login",
    ],
    [{ provider: "anthropic", project: "proj-1" }, "organization and project are only for an openai API-key login"],
    [{ method: "subscription", headerNames: ["X-A"] }, "Connection headers are only for an API-key login"],
    [{ method: "none", headerNames: ["X-A"] }, "Connection headers are only for an API-key login"],
    [{ provider: "ollama", headerNames: ["X-A"] }, "Connection headers are not sent to provider ollama"],
    [
      { organization: "has space" },
      "Invalid provider selection: organization must be non-empty visible ASCII, without spaces",
    ],
    [
      { headerNames: ["Authorization"] },
      "Invalid provider selection: header Authorization is refused: yian sends the credential itself",
    ],
    [
      { headerNames: ["OpenAI-Project"] },
      "Invalid provider selection: header OpenAI-Project is refused: use llm_config.organization / project",
    ],
    [
      { headerNames: ["X-A", "x-a"] },
      "Invalid provider selection: a header name is given more than once (names ignore case)",
    ],
    [
      { headerNames: Array.from({ length: 33 }, (_, n) => `X-${n}`) },
      "Invalid provider selection: headers must have at most 32 headers",
    ],
  ];
  for (const [fields, message] of refusals) {
    test(`refuses ${JSON.stringify(fields).slice(0, 60)}`, () => {
      const values = { provider: "openai", method: "api_key", model: "m", credentialId: "c".repeat(32), ...fields };
      const error = thrown(() => new ProviderSelection(values));
      expect(error).toBeInstanceOf(AuthError);
      expect(error.message).toBe(message);
    });
  }
});

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
    const directory = join(path, "..");
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

  test("a v1 store without the new fields still loads", () => {
    writeRaw(
      file,
      '{"default_provider": "openai", "providers": {"openai": {"credential_id": ' +
        '"0123456789abcdef0123456789abcdef", "credentials": {"api_key": "sk-old"}, ' +
        '"method": "api_key", "model": "gpt-4o"}}, "version": 1}\n',
    );
    const selection = new AuthStore(file).selection();
    expect(selection?.provider).toBe("openai");
    expect(selection?.method).toBe("api_key");
    expect(selection?.endpoint).toBeNull();
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
});
