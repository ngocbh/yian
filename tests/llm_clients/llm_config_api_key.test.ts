/**
 * API keys in `LLMConfig` and `chatModelFromConfig`: empty keys, keys with control characters and
 * the host key marker, host intake through `LLMConfig.fromHost`, model config resolution,
 * connection credentials, saved subscriptions, and the auth headers each client sends (a fake
 * `send` answers 400, so a test checks the request). `describe("redact")` redacts keys from text;
 * `describe("scrub")` scrubs them from errors, through `cause` chains, `AggregateError` members
 * and string members. Validation messages never hold a value.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { REDACTED as AUTH_REDACTED } from "../../src/llm/auth.ts";
import { ProviderError, scrubError } from "../../src/llm/errors.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { AuthError, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import {
  type Connection,
  connectionCredentials,
  resolveModelConfig,
  splitModelConfig,
  validateConnection,
} from "../../src/llm_clients/connection.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";
import { REDACTED, redactKey, redactKeys } from "../../src/llm_clients/redact.ts";
import { type Send, virtualScheduler } from "../../src/platform.ts";

const SENTINEL = "sk-sentinel-3f9c1a7e";

// Each test's environment: a private store in a fresh directory and no key or default.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-api-key-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the error that `make` throws, which must be a `ValidationError`. */
function refusal(make: () => unknown): ValidationError {
  try {
    make();
  } catch (error) {
    expect(error).toBeInstanceOf(ValidationError);
    return error as ValidationError;
  }
  throw new Error("expected a ValidationError");
}

/** Return the error that building an `LLMConfig` from `data` throws. */
function thrown(data: Record<string, unknown>): ValidationError {
  return refusal(() => new LLMConfig(data));
}

test("inspect, String and dumps hide the key", () => {
  const config = new LLMConfig({ provider: "anthropic", model: "claude-sonnet-4-5", api_key: SENTINEL });
  expect(config.apiKey).toBe(SENTINEL);
  for (const text of [inspect(config), String(config), inspect(config.dump()), JSON.stringify(config)]) {
    expect(text).not.toContain(SENTINEL);
  }
  expect(Object.hasOwn(config.dump(), "api_key")).toBe(false);
});

test("the key field does not change dumps of configs without it", () => {
  // Recorded snapshots compare full dumps, so an unset key must add nothing.
  expect(Object.hasOwn(new LLMConfig({ provider: "openai", model: "gpt-5.5" }).dump(), "api_key")).toBe(false);
});

test.each([
  [{ openai: { temperature: "hot" } }],
  [{ temparature: 0.2 }],
  [{ provider: "nope" }],
  [{ api_key: 123n }],
])("validation errors do not echo config values %#", (fields: Record<string, unknown>) => {
  const error = thrown({ api_key: SENTINEL, ...fields });
  expect(error.message).not.toContain(SENTINEL);
  expect(inspect(error)).not.toContain(SENTINEL);
  for (const value of Object.values(fields)) {
    const leaves = typeof value === "object" && value !== null ? Object.values(value) : [value];
    for (const leaf of leaves) {
      expect(error.message.replace("temparature", "")).not.toContain(String(leaf));
    }
  }
});

describe("fromHost", () => {
  const KEY = "sk-host-key-0123456789";

  test("fromHost takes a JSON config with whole numbers as ints", () => {
    const data = { provider: "anthropic", api_key: KEY, anthropic: { max_tokens: 1024 } };
    const config = LLMConfig.fromHost(data);
    expect(config.blockFor("anthropic")["max_tokens"]).toBe(1024n);
    expect(config.apiKey).toBe(KEY);
    expect(thrown(data).issues.map((issue) => issue.type)).toContain("int_type");
  });

  test("fromHost refuses what the constructor refuses", () => {
    const data = { provider: "nope" };
    expect(refusal(() => LLMConfig.fromHost(data)).issues).toEqual(thrown(data).issues);
  });

  test("fromHost refuses a value that is not a yian value", () => {
    const key = "sk-proj-SENTINEL-0123456789abcdef";
    for (const data of [{ model: NaN }, JSON.parse('{"model": "\\ud800"}'), { api_key: `${key}\ud800` }]) {
      let error: unknown;
      try {
        LLMConfig.fromHost(data);
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).name).toBe("TypeCheckError");
      expect((error as Error).message).not.toContain("SENTINEL");
    }
  });
});

describe("value-free validation messages", () => {
  const K = "sk-proj-SENTINEL-0123456789abcdef";
  const KEYED_OPENAI = { provider: "openai", api_key: K };
  // Each config is refused and holds K as a value, never as a key.
  const CASES: [string, Record<string, unknown>][] = [
    ["provider", { provider: K }],
    ["api_key beside a bad model", { model: 5, api_key: K }],
    ["base_url userinfo", { ...KEYED_OPENAI, base_url: `https://${K}@x.example/` }],
    ["base_url query", { ...KEYED_OPENAI, base_url: `https://x.example/?k=${K}` }],
    ["api_key without provider", { api_key: K, base_url: "https://x.example/v1" }],
    ["organization", { organization: K + " x" }],
    ["project", { project: K + "\n" }],
    ["refused header name", { headers: { Authorization: K } }],
    ["header value", { headers: { "X-Gateway": K + "\n" } }],
    ["reasoning effort", { openai: { reasoning: { effort: K } } }],
    ["reasoning", { openai: { reasoning: K } }],
    ["long safety_identifier", { openai: { safety_identifier: K.repeat(4) } }],
    ["long metadata value", { openai: { metadata: { a: K.repeat(20) } } }],
    ["temperature", { temperature: K }],
    ["retired anthropic_effort", { anthropic_effort: K }],
    ["anthropic block", { anthropic: { output_config: { effort: K } } }],
    ["google block", { google: { generationConfig: { thinkingConfig: { thinkingLevel: K } } } }],
    ["sandbox", { sandbox: K }],
  ];

  test.each(CASES)("validation messages and issues never hold a key-shaped value (%s)", (_label, data) => {
    for (const make of [() => new LLMConfig(data), () => LLMConfig.fromHost(data)]) {
      const error = refusal(make);
      expect(error.message).not.toContain("SENTINEL");
      expect(JSON.stringify(error.issues)).not.toContain("SENTINEL");
    }
  });

  test("validation messages name keys as written, at any depth", () => {
    for (const data of [{ [K]: 1 }, { openai: { [K]: 1 } }, { openai: { metadata: { [K]: 5 } } }]) {
      expect(thrown(data).message).toContain(K);
    }
  });
});

/** Build `config` with this test's environment, keeping warnings quiet. */
function build(config: Record<string, unknown>, send?: Send) {
  return chatModelFromConfig(config, { env, send, scheduler: virtualScheduler(), onWarning: () => {} });
}

/** Return what `promise` rejects with; `null` if it resolves. */
function rejection(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(
    () => null,
    (thrown: unknown) => thrown,
  );
}

interface Sent {
  url: string;
  headers: Headers;
}

/** A `send` that records each request and answers it 400. */
function recorder(): { requests: Sent[]; send: Send } {
  const requests: Sent[] = [];
  const send: Send = async (input, init) => {
    requests.push({ url: String(input), headers: new Headers(init.headers) });
    return new Response(JSON.stringify({ error: { message: "refused" } }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };
  return { requests, send };
}

const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

/** Build `config` with a recording `send`, ask it "hi", and return the requests sent. */
async function requestsOf(config: Record<string, unknown>): Promise<Sent[]> {
  const { requests, send } = recorder();
  const model = await build(config, send);
  await rejection(collect(model.stream(HI)));
  return requests;
}

test("an empty key is rejected instead of falling back", async () => {
  vars.set("GOOGLE_API_KEY", "sk-environment-key");
  expect(thrown({ provider: "google", api_key: "" }).message).toContain("api_key");
  const error = await rejection(build({ provider: "google", model: "gemini-2.5-flash", api_key: "" }));
  expect(error).toBeInstanceOf(ValidationError);
  expect((error as Error).message).toContain("api_key");
});

test.each([
  ["LF", "\n"],
  ["CR", "\r"],
  ["tab", "\t"],
  ["NUL", "\u0000"],
  ["U+001F", "\u001f"],
  ["DEL", "\u007f"],
])("a key with a %s is refused without echoing it", (_name, control) => {
  const error = thrown({ provider: "openai", api_key: `sk-dummy${control}zz` });
  expect(error.issues).toEqual([
    {
      loc: ["api_key"],
      type: "value_error",
      msg: "Value error, api_key must not contain a control character",
      error: "api_key must not contain a control character",
    },
  ]);
  expect(error.message).not.toContain("sk-dummy");
});

test("a key with inner spaces or non-ASCII is kept", () => {
  for (const key of ["sk dummy", "sk-dummy-\u00e9"]) {
    expect(new LLMConfig({ provider: "openai", api_key: key }).apiKey).toBe(key);
  }
});

test.each([
  ["vllm", "served-model"],
  ["anthropic", "claude-sonnet-4-5"],
])("a host key marker without the key fails before the environment (%s)", async (provider, model) => {
  vars.set("VLLM_API_KEY", "sk-environment-key");
  vars.set("ANTHROPIC_API_KEY", "sk-environment-key");
  const marked = { provider, model, _host_key: true };
  const built = await rejection(build(marked));
  expect(built).toBeInstanceOf(AuthError);
  expect((built as AuthError).message).toMatch(/llm_config\.api_key/);
  let error: unknown;
  try {
    splitModelConfig(marked);
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(AuthError);
  expect((error as AuthError).message).toMatch(/llm_config\.api_key/);
  const [validated] = splitModelConfig({ ...marked, api_key: SENTINEL });
  expect(validated.apiKey).toBe(SENTINEL);
  expect(() => splitModelConfig({ ...marked, _host_key: "yes", api_key: SENTINEL })).toThrow(/marker/);
});

test("a marked config's headers that are not a record are refused as malformed, not as unrecorded", () => {
  for (const headers of [[], "", false, 0n]) {
    const marked = {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      api_key: SENTINEL,
      headers,
      _host_key: true,
      _header_names: ["X-Gateway"],
    };
    const error = refusal(() => splitModelConfig(marked));
    expect(error.issues.map(({ loc, type }) => [loc, type])).toEqual([[["headers"], "dict_type"]]);
  }
});

test("split and resolve keep the key as a secret", () => {
  const config = new LLMConfig({ provider: "anthropic", model: "claude-sonnet-4-5", api_key: SENTINEL });
  const [validated] = splitModelConfig(config);
  expect(validated.apiKey).toBe(SENTINEL);
  // The resolved record carries the key as plain text for the factory; only `LLMConfig` hides it.
  const resolved = resolveModelConfig({ ...config.dump(), api_key: SENTINEL }, { env });
  expect(resolved["api_key"]).toBe(SENTINEL);
});

test("openai config key wins a saved subscription", async () => {
  await new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: 4102444800,
      accountId: "account",
    }),
    "gpt-5.5",
  );
  const resolved = resolveModelConfig({ provider: "openai", api_key: SENTINEL }, { env });
  const connection = resolved["_connection"] as Connection;
  expect(connection.source).toBe("host");
  expect(connection.credential_id).toBeNull();
  expect(JSON.stringify(connection)).not.toContain(SENTINEL);
  const requests = await requestsOf(resolved);
  expect(new URL(requests[0]!.url).host).toBe("api.openai.com");
  expect(requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
});

// --- clients prefer the config key --- //

// The key header each provider's route sends.
const KEYED: [string, string, string, string, string][] = [
  ["openai", "gpt-5.5", "OPENAI_API_KEY", "Authorization", `Bearer ${SENTINEL}`],
  ["vllm", "served-model", "VLLM_API_KEY", "Authorization", `Bearer ${SENTINEL}`],
  ["deepseek", "deepseek-chat", "DEEPSEEK_API_KEY", "Authorization", `Bearer ${SENTINEL}`],
  ["openrouter", "some/model", "OPENROUTER_API_KEY", "Authorization", `Bearer ${SENTINEL}`],
  ["anthropic", "claude-sonnet-4-5", "ANTHROPIC_API_KEY", "x-api-key", SENTINEL],
  ["google", "gemini-2.5-flash", "GOOGLE_API_KEY", "x-goog-api-key", SENTINEL],
  ["azure", "gpt-4o", "AZURE_OPENAI_API_KEY", "api-key", SENTINEL],
];

for (const envValue of [null, "sk-environment-key"]) {
  test.each(KEYED)(
    `every keyed client prefers the config key (%s, env ${envValue ?? "unset"})`,
    async (provider, model, variable, header, value) => {
      const config: Record<string, unknown> = { provider, model, api_key: SENTINEL };
      if (provider === "azure") {
        vars.set("OPENAI_API_VERSION", "2024-02-01");
        config["base_url"] = "https://res.openai.azure.com/";
      }
      if (envValue !== null) vars.set(variable, envValue);
      const chat = await build(config);
      expect(await chat.route.auth.headers()).toEqual({ [header]: value });
    },
  );
}

test("openai-compatible endpoint receives the config key", async () => {
  vars.set("VLLM_BASE_URL", "http://127.0.0.1:9/v1");
  const requests = await requestsOf({ provider: "vllm", model: "served-model", api_key: SENTINEL });
  expect(requests.map(({ url }) => url)).toEqual(["http://127.0.0.1:9/v1/chat/completions"]);
  expect(requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
});

test("host connection needs the config key", async () => {
  const connection = validateConnection({
    method: "api_key",
    source: "host",
    credential_id: null,
    endpoint: "https://api.openai.com/v1",
    store_path: null,
  });
  let error: unknown;
  try {
    await connectionCredentials(connection, "openai", { env });
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(AuthError);
  expect((error as AuthError).message).toContain("api_key");
  expect(() => validateConnection({ ...connection, credential_id: "a".repeat(64) })).toThrow(RangeError);
  expect(() => validateConnection({ ...connection, method: "subscription" })).toThrow(RangeError);
});

describe("redact", () => {
  test("the key and its masked fragments leave a chain's messages", () => {
    const key = "sk-sentinel-scrub-zqxw";
    const messages = [
      `Incorrect API key provided: ${key.slice(0, 8)}****${key.slice(-4)}.`,
      `wrapped: ${key}`,
      key,
    ].map((message) => redactKey(message, key));
    const text = messages.join("\n");
    for (const fragment of [key, key.slice(0, 8), "*" + key.slice(-4)]) {
      expect(text).not.toContain(fragment);
    }
    expect(messages).toEqual(["Incorrect API key provided: [REDACTED].", "wrapped: [REDACTED]", "[REDACTED]"]);
  });

  test("unrelated masks are left alone", () => {
    expect(redactKey("token ab****yz and **bold**", "sk-sentinel-zqxw")).toBe("token ab****yz and **bold**");
    expect(redactKey("anything", null)).toBe("anything");
  });

  test.each([
    ["run", "f".repeat(100_000)],
    ["single-stars", "ab*".repeat(33_000)],
    ["hex-blob", "0123456789abcdef".repeat(6_250) + "!"],
    ["dots", "a.".repeat(50_000)],
  ])("redaction is linear on long text: %s", (_name: string, text: string) => {
    const started = performance.now();
    redactKey(text, "sk-sentinel-scrub-zqxw");
    expect(performance.now() - started).toBeLessThan(500);
  });

  test("masked echoes need three visible characters", () => {
    const key = "sk-sentinel-scrub-zqxz";
    expect(redactKey("s**t, **z, and sk**", key)).toBe("s**t, **z, and sk**");
    expect(redactKey("sk***, ***xz, and ****qxz", key)).toBe("sk***, ***xz, and [REDACTED]");
    expect(redactKey("sk-***", key)).toBe("[REDACTED]");
  });

  test("masked echoes of a key with other characters are redacted whole", () => {
    const key = "sk.sentinel+scrub/zq.w";
    for (const echo of [`${key.slice(0, 8)}****${key.slice(-4)}`, `${key.slice(0, 9)}...`, `***${key.slice(-5)}`]) {
      expect(redactKey(`provided: ${echo}.`, key)).toBe("provided: [REDACTED].");
    }
  });

  test("each key leaves a chain's messages", () => {
    // Distinct prefixes and suffixes, so each masked echo matches only its own key.
    const first = "ak-first-sentinel-zqxw";
    const second = "sk-second-sentinel-yvkt";
    const messages = [
      `Incorrect API key provided: ${second.slice(0, 8)}****${second.slice(-4)}. Also ${first.slice(0, 8)}...${first.slice(-4)}`,
      `wrapped: ${first}`,
    ].map((message) => redactKeys(message, [first, second, null, undefined, ""]));
    const text = messages.join("\n");
    for (const fragment of [first, second, first.slice(0, 8), second.slice(0, 8), "." + first.slice(-4), "*" + second.slice(-4)]) {
      expect(text).not.toContain(fragment);
    }
    expect(messages).toEqual(["Incorrect API key provided: [REDACTED]. Also [REDACTED]", "wrapped: [REDACTED]"]);
  });

  test("a containing key is redacted whole", () => {
    const text = redactKeys("rejected abcdefgh-longer-1234 here", ["abcdefgh", "abcdefgh-longer-1234"]);
    expect(text).toBe("rejected [REDACTED] here");
    expect(text).not.toContain("-longer-1234");
  });

  test("encoded and masked echoes of keys with /, a space, ~, ], \\, [ and ^ are removed", () => {
    const key = "sk/sent inel~scr]ub-zqxw";
    const json = JSON.stringify(key).slice(1, -1).replaceAll("/", "\\/");
    expect(json).toBe("sk\\/sent inel~scr]ub-zqxw");
    expect(redactKey(`body: {"key": "${json}"}`, key)).toBe(`body: {"key": "[REDACTED]"}`);
    expect(redactKey("GET /v1?key=sk%2Fsent%20inel~scr%5Dub-zqxw", key)).toBe("GET /v1?key=[REDACTED]");
    expect(redactKey(`provided: ${key.slice(0, 8)}****${key.slice(-4)}.`, key)).toBe("provided: [REDACTED].");

    // The masked head and tail both hold characters a character class must escape.
    const other = "sk\\se[nt^inel-z\\q^w";
    expect(redactKey(`plain: ${other}`, other)).toBe("plain: [REDACTED]");
    expect(redactKey(`body: {"key": "sk\\\\se[nt^inel-z\\\\q^w"}`, other)).toBe(`body: {"key": "[REDACTED]"}`);
    expect(redactKey("GET /v1?key=sk%5Cse%5Bnt%5Einel-z%5Cq%5Ew", other)).toBe("GET /v1?key=[REDACTED]");
    expect(redactKey(`provided: ${other.slice(0, 8)}****${other.slice(-4)}.`, other)).toBe("provided: [REDACTED].");
  });

  test("masked echoes with ... and … are removed", () => {
    const key = "sk-sentinel-scrub-zqxw";
    expect(redactKey(`a ${key.slice(0, 9)}... b …${key.slice(-4)} c ${key.slice(0, 6)}…`, key)).toBe(
      "a [REDACTED] b [REDACTED] c [REDACTED]",
    );
  });

  test("a key with astral characters counts visible characters in code points", () => {
    const key = "sk-sentinel-zq\u{1F600}\u{1F389}";
    // Two astral characters are four UTF-16 units but only two visible characters.
    expect(redactKey("x ***\u{1F600}\u{1F389} y", key)).toBe("x ***\u{1F600}\u{1F389} y");
    expect(redactKey("x ***q\u{1F600}\u{1F389} y", key)).toBe("x [REDACTED] y");
    expect(redactKey("x sk-senti**** y", key)).toBe("x [REDACTED] y");
    expect(redactKey("x \u{1F600}\u{1F389}****", "\u{1F600}\u{1F389}abc")).toBe("x \u{1F600}\u{1F389}****");
    expect(redactKey("x \u{1F600}\u{1F389}a****", "\u{1F600}\u{1F389}abc")).toBe("x [REDACTED]");
  });

  test("null, undefined and empty keys are ignored", () => {
    const text = "provided: sk-***";
    for (const key of [null, undefined, ""]) expect(redactKey(text, key)).toBe(text);
    expect(redactKeys(text, [null, undefined, ""])).toBe(text);
    expect(redactKey("", "sk-sentinel")).toBe("");
  });

  test("REDACTED is llm/auth.ts's marker", () => {
    expect(REDACTED).toBe(AUTH_REDACTED);
  });
});

/**
 * Return the stacks of `error` and each of its causes, as a traceback shows a chain. Bun
 * builds a stack on its first read, so tests read the stacks before scrubbing too, as a logger may
 * have; the scrub then has to rewrite them.
 */
function traceback(error: unknown): string {
  const parts: string[] = [];
  for (let current = error; current instanceof Error; current = current.cause) parts.push(String(current.stack));
  return parts.join("\n");
}

describe("scrub", () => {
  const KEY = "sk-sentinel-scrub-zqxw";

  test("the key and its masked fragments leave a chain", () => {
    const inner = new Error(`Incorrect API key provided: ${KEY.slice(0, 8)}****${KEY.slice(-4)}.`);
    const error = new Error(KEY, { cause: new Error(`wrapped: ${KEY}`, { cause: inner }) });
    expect(traceback(error)).toContain(KEY);
    scrubError(error, (text) => redactKey(text, KEY));
    const text = traceback(error);
    for (const fragment of [KEY, KEY.slice(0, 8), "*" + KEY.slice(-4)]) expect(text).not.toContain(fragment);
    expect(text).toContain("Incorrect API key provided: [REDACTED].");
    expect(text).toContain("wrapped: [REDACTED]");
  });

  test("aggregate members are reached", () => {
    const member = new Error(`member ${KEY}`);
    const group = new AggregateError([member, `note ${KEY.slice(0, 8)}***`], "group");
    expect(traceback(group) + traceback(member)).toContain(KEY);
    scrubError(group, (text) => redactKey(text, KEY));
    const text = [traceback(group), traceback(member), group.errors[1]].join("\n");
    expect(text).not.toContain(KEY);
    expect(text).not.toContain(KEY.slice(0, 8));
    expect(group.errors[1]).toBe("note [REDACTED]");
  });

  test("errors it cannot rewrite are skipped", () => {
    const deepest = new Error(`deepest ${KEY}`);
    const frozen = Object.freeze(new Error("frozen", { cause: deepest }));
    const broken = new Error(`broken ${KEY}`, { cause: frozen });
    // The stack is built before the message breaks, so it holds the key.
    expect(broken.stack).toContain(KEY);
    Object.defineProperty(broken, "message", {
      get() {
        throw new Error("no message");
      },
    });
    const strings = Object.assign(new Error(`strings ${KEY}`, { cause: broken }), { errors: { member: `text ${KEY}` } });
    const number = Object.assign(new Error(`number ${KEY}`, { cause: strings }), { errors: 5 });
    expect(traceback(number)).toContain(KEY);
    scrubError(number, (text) => redactKey(text, KEY));
    for (const error of [number, strings, deepest]) {
      expect(error.message).not.toContain(KEY);
      expect(error.message).toContain(REDACTED);
    }
    for (const error of [number, strings, broken, deepest]) {
      expect(String(error.stack)).not.toContain(KEY);
      expect(String(error.stack)).toContain(REDACTED);
    }
  });

  test("a ProviderError message is rewritten", () => {
    const error = new ProviderError("auth", `Incorrect API key provided: ${KEY.slice(0, 8)}****${KEY.slice(-4)}.`);
    scrubError(error, (text) => redactKey(text, KEY));
    expect(error.message).toBe("Incorrect API key provided: [REDACTED].");
  });

  test("each key leaves a chain", () => {
    // Distinct prefixes and suffixes, so each masked echo matches only its own key.
    const first = "ak-first-sentinel-zqxw";
    const second = "sk-second-sentinel-yvkt";
    const inner = new Error(
      `Incorrect API key provided: ${second.slice(0, 8)}****${second.slice(-4)}. Also ${first.slice(0, 8)}...${first.slice(-4)}`,
    );
    const error = new Error(`wrapped: ${first}`, { cause: inner });
    expect(traceback(error)).toContain(first);
    scrubError(error, (text) => redactKeys(text, [first, second, null, undefined, ""]));
    const text = traceback(error);
    for (const fragment of [first, second, first.slice(0, 8), second.slice(0, 8), "." + first.slice(-4), "*" + second.slice(-4)]) {
      expect(text).not.toContain(fragment);
    }
    expect(text).toContain("wrapped: [REDACTED]");
    expect(text).toContain("Incorrect API key provided: [REDACTED]. Also [REDACTED]");
  });

  test("a containing key is redacted whole", () => {
    const [short, long] = ["abcdefgh", "abcdefgh-longer-1234"];
    const error = new Error(`rejected ${long} here`);
    expect(error.stack).toContain(long);
    scrubError(error, (text) => redactKeys(text, [short, long]));
    expect(error.message).toBe("rejected [REDACTED] here");
    expect(String(error.stack)).not.toContain("-longer-1234");
  });
});
