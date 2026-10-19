/** Tests for the per-model capability table in `src/llm/capabilities.ts`. */

import { describe, expect, test } from "bun:test";

import { getCapabilities, supportsNativeStructured } from "../../src/llm/capabilities.ts";

describe("getCapabilities", () => {
  test("deepseek-reasoner rejects tool choice", () => {
    const caps = getCapabilities("deepseek-reasoner");
    expect(caps.supportsToolChoice).toBe(false);
    expect(caps.requiresReasoningContentRoundtrip).toBe(true);
  });

  test("MiniMax requires reasoning_split", () => {
    expect(getCapabilities("MiniMax-M2.7").requiresReasoningSplit).toBe(true);
  });

  test("default model allows tool choice", () => {
    const caps = getCapabilities("gpt-4o");
    expect(caps.supportsToolChoice).toBe(true);
    expect(caps.preferredStructuredMethod).toBe("function_calling");
  });

  test("forward-compat pattern", () => {
    // An unseen deepseek-v5 or MiniMax-M3 variant inherits the thinking-mode quirks.
    expect(getCapabilities("deepseek-v5-pro").supportsToolChoice).toBe(false);
    expect(getCapabilities("MiniMax-M3").requiresReasoningSplit).toBe(true);
  });

  test("deepseek-chat allows tool choice", () => {
    expect(getCapabilities("deepseek-chat").supportsToolChoice).toBe(true);
  });

  test("exact ids and patterns share profiles", () => {
    expect(getCapabilities("deepseek-v4-pro")).toBe(getCapabilities("deepseek-reasoner"));
    expect(getCapabilities("MiniMax-M2")).toBe(getCapabilities("MiniMax-M9"));
  });

  test("the catalog sentinel has no structured method", () => {
    expect(getCapabilities("no-structured-sentinel").preferredStructuredMethod).toBe("none");
  });

  test("profiles are frozen", () => {
    for (const model of ["deepseek-chat", "deepseek-reasoner", "MiniMax-M2.7", "gpt-4o", "no-structured-sentinel"]) {
      expect(Object.isFrozen(getCapabilities(model))).toBe(true);
    }
  });
});

describe("supportsNativeStructured", () => {
  test("anthropic, openai and google support native", () => {
    expect(supportsNativeStructured("anthropic", "claude-sonnet-4-5")).toBe(true);
    expect(supportsNativeStructured("openai", "gpt-5.5")).toBe(true);
    expect(supportsNativeStructured("google", "gemini-3-pro")).toBe(true);
  });

  test("known no-native model returns false", () => {
    // The catalog sentinel: the one entry whose method is `none`.
    expect(supportsNativeStructured("vllm", "no-structured-sentinel")).toBe(false);
  });

  test("unknown defaults to native", () => {
    expect(supportsNativeStructured("some-future-provider", "some-model")).toBe(true);
  });

  test("provider is compared case-insensitively", () => {
    expect(supportsNativeStructured("AZURE", "no-structured-sentinel")).toBe(true);
  });
});
