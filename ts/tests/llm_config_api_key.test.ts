/**
 * Mirrors `tests/test_llm_config_api_key.py`: the 4 `LLMConfig` tests, the empty-key test without
 * its `model_from_config` half (that half goes with the factory), and the host-key marker test
 * through `splitModelConfig`. `resolve_model_config`, `connection_credentials` and
 * `validate_connection` go with `connection.ts`; the client tests with the factory. The
 * `redact_key(s)` text tests are here, in `describe("redact")`, with the text half of the
 * `scrub_key(s)` chain tests; the exception-walk tests go with `scrubError` in `llm/errors.ts`.
 */

import { inspect } from "node:util";

import { describe, expect, test } from "bun:test";

import { REDACTED as AUTH_REDACTED } from "../src/llm/auth.ts";
import { AuthError } from "../src/llm_clients/auth/common.ts";
import { LLMConfig } from "../src/llm_clients/config.ts";
import { splitModelConfig } from "../src/llm_clients/connection.ts";
import { ValidationError } from "../src/llm_clients/provider_blocks.ts";
import { REDACTED, redactKey, redactKeys } from "../src/llm_clients/redact.ts";

const SENTINEL = "sk-sentinel-3f9c1a7e";

/** Return the error that building an `LLMConfig` from `data` throws. */
function thrown(data: Record<string, unknown>): ValidationError {
  try {
    new LLMConfig(data);
  } catch (error) {
    expect(error).toBeInstanceOf(ValidationError);
    return error as ValidationError;
  }
  throw new Error("expected a ValidationError");
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

test("an empty key is rejected instead of falling back", () => {
  expect(thrown({ provider: "google", api_key: "" }).message).toContain("api_key");
});

test.each([
  ["vllm", "served-model"],
  ["anthropic", "claude-sonnet-4-5"],
])("a host key marker without the key fails before the environment (%s)", (provider, model) => {
  const marked = { provider, model, _host_key: true };
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

  test("encoded and masked echoes of a key with /, a space, ~ and ] are removed", () => {
    const key = "sk/sent inel~scr]ub-zqxw";
    const json = JSON.stringify(key).slice(1, -1).replaceAll("/", "\\/");
    expect(json).toBe("sk\\/sent inel~scr]ub-zqxw");
    expect(redactKey(`body: {"key": "${json}"}`, key)).toBe(`body: {"key": "[REDACTED]"}`);
    expect(redactKey("GET /v1?key=sk%2Fsent%20inel~scr%5Dub-zqxw", key)).toBe("GET /v1?key=[REDACTED]");
    expect(redactKey(`provided: ${key.slice(0, 8)}****${key.slice(-4)}.`, key)).toBe("provided: [REDACTED].");
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
