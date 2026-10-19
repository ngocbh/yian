/**
 * Mirrors `tests/test_llm_config_api_key.py`: the 4 `LLMConfig` tests, the empty-key test without
 * its `model_from_config` half (that half goes with the factory). `split_model_config`,
 * `resolve_model_config`, `connection_credentials` and `validate_connection` go with
 * `connection.ts`; `scrub_key(s)` and `redact_key` with `redact.ts`; the client tests with the
 * factory.
 */

import { inspect } from "node:util";

import { expect, test } from "bun:test";

import { LLMConfig } from "../src/llm_clients/config.ts";
import { ValidationError } from "../src/llm_clients/provider_blocks.ts";

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
