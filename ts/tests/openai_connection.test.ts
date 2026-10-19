/**
 * Mirrors `tests/test_openai_connection.py`; model calls go with the factory,
 * `llm_clients/factory.ts`; runs with `runFlow`; the cascade with `compile/llm_cascade.ts`; the key
 * preflight with the CLI.
 */

import { expect, test } from "bun:test";

import { LLMConfig } from "../src/llm_clients/config.ts";
import { legacyConnectionConfig, splitModelConfig } from "../src/llm_clients/connection.ts";

test("internal metadata is not YAML configuration", () => {
  // An OpenAI config with its pinned connection, as a run record carries it.
  const config = legacyConnectionConfig({ provider: "openai", model: "gpt-5.5" });
  expect(() => new LLMConfig(config)).toThrow(/Extra inputs/);
  (config["_connection"] as Record<string, unknown>)["token"] = "bad";
  expect(() => splitModelConfig(config)).toThrow(/connection/);
});
