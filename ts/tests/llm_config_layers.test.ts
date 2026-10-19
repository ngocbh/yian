/**
 * Mirrors `tests/test_llm_config_layers.py`; the comparison with `resolve_llm_cascade` waits for
 * `compile/llm_cascade.ts`.
 */

import { describe, expect, test } from "bun:test";

import { HOST_KEY_MARKER, missingModelMessage } from "../src/llm_clients/connection.ts";
import { requireConcreteSelection, resolveAgentConfig, validateLayer } from "../src/llm_clients/layers.ts";

const DEFAULTS = { provider: "anthropic", model: "default-model" };

/** Resolve `authored` as the agent at `review/chat`. */
function resolve(
  authored: Record<string, unknown>,
  options: {
    defaults?: Record<string, unknown>;
    providerModels?: Record<string, string>;
    strayModel?: boolean;
    label?: string | null;
  } = {},
): Record<string, unknown> {
  return resolveAgentConfig(authored, {
    defaults: options.defaults ?? DEFAULTS,
    providerModels: options.providerModels ?? {},
    strayModel: options.strayModel ?? false,
    path: "review/chat",
    label: options.label ?? null,
  });
}

/** Return the message of the `RangeError` `run` throws, failing the test on any other outcome. */
function rangeError(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(RangeError);
    return (error as RangeError).message;
  }
  throw new Error("expected a RangeError");
}

test("only the default provider gives the default model", () => {
  expect(resolve({})).toEqual({ provider: "anthropic", model: "default-model" });
});

test("defaults other than model fill gaps under the authored layer", () => {
  const defaults = { ...DEFAULTS, anthropic: { temperature: 0.1, max_tokens: 10n } };

  const config = resolve({ anthropic: { temperature: 0.5 } }, { defaults });

  expect(config).toEqual({
    provider: "anthropic",
    model: "default-model",
    anthropic: { temperature: 0.5, max_tokens: 10n },
  });
});

test("an authored model wins over saved and default models", () => {
  const config = resolve({ provider: "openai", model: "gpt-authored" }, { providerModels: { openai: "gpt-saved" } });

  expect(config).toEqual({ provider: "openai", model: "gpt-authored" });
});

test("a saved model serves an authored provider", () => {
  const config = resolve({ provider: "openai" }, { providerModels: { openai: "gpt-saved" } });

  expect(config).toEqual({ provider: "openai", model: "gpt-saved" });
});

test("the default model never serves another provider", () => {
  expect(rangeError(() => resolve({ provider: "openai" }))).toBe(
    "Agent 'review/chat' has no model for provider 'openai'; set " +
      "llm_config.provider/model, pass --provider/--model, or run 'yian auth openai'",
  );
});

test("a missing model names the label", () => {
  const message = rangeError(() =>
    resolve({ provider: "openai" }, { label: "adaptive_questions.llm_config of node 'ask'" }),
  );

  expect(message.startsWith("Adaptive_questions.llm_config of node 'ask' has no model for provider 'openai'")).toBe(
    true,
  );
});

test("a missing provider names the path", () => {
  const message = rangeError(() => resolve({}, { defaults: {} }));

  expect(message.startsWith("Agent 'review/chat' has no provider or model;")).toBe(true);
});

test("a stray default model is named", () => {
  const message = rangeError(() => resolve({}, { defaults: {}, strayModel: true }));

  expect(message).toBe(missingModelMessage("review/chat", null, { strayModel: true }));
  expect(message).toContain("YIAN_DEFAULT_MODEL");
});

test.each([
  [HOST_KEY_MARKER, true, "internal to recorded runs"],
  ["base_url", "https://example.test/v1", "base_url"],
])("a host marker or base_url is rejected: %s", (field, value, message) => {
  expect(rangeError(() => resolve({ provider: "openai", model: "m", [field]: value }))).toContain(message);
});

test("the result never aliases its inputs", () => {
  const authored = { provider: "openai", openai: { temperature: 0.3, max_output_tokens: 5n } };

  const config = resolve(authored, { providerModels: { openai: "gpt-saved" } });

  expect(config).toEqual({ provider: "openai", model: "gpt-saved", openai: { temperature: 0.3, max_output_tokens: 5n } });
  (config["openai"] as Record<string, unknown>)["temperature"] = 9;
  expect(authored.openai.temperature).toBe(0.3);
});

describe("TS additions", () => {
  test("a non-string default provider takes the default model, then is not concrete", () => {
    expect(rangeError(() => resolve({}, { defaults: { provider: 1n, model: "m" } }))).toBe(
      'effective LLM config for \'review/chat\' lacks concrete fields: ["provider"]',
    );
  });

  test("a list provider never matches a model", () => {
    const provider = ["openai"];
    expect(rangeError(() => resolve({}, { defaults: { provider, model: "m" } }))).toBe(
      missingModelMessage("review/chat", provider),
    );
  });

  test("the exact messages", () => {
    expect(rangeError(() => requireConcreteSelection({ provider: "openai" }, { path: "agent" }))).toBe(
      'effective LLM config for \'agent\' lacks concrete fields: ["model"]',
    );
    expect(rangeError(() => validateLayer({ _header_names: [] }))).toBe(
      "llm_config._header_names is internal to recorded runs",
    );
  });

  test("undefined values are absent", () => {
    requireConcreteSelection(
      {
        provider: "openai",
        model: "m",
        [HOST_KEY_MARKER]: undefined,
        base_url: undefined,
        organization: undefined,
        project: undefined,
        _header_names: undefined,
      },
      { path: "agent" },
    );
    const config = resolve(
      { provider: "openai", model: undefined, [HOST_KEY_MARKER]: undefined, _header_names: undefined },
      { providerModels: { openai: "gpt-saved" } },
    );
    expect(config).toEqual({ provider: "openai", model: "gpt-saved" });
  });
});
