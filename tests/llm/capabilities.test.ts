/** Tests for the model profile table in `src/llm/capabilities.ts`. */

import { describe, expect, test } from "bun:test";

import { modelProfile, supportsNativeStructured } from "../../src/llm/capabilities.ts";
import { getKnownModels } from "../../src/llm_clients/model_catalog.ts";

describe("modelProfile", () => {
  const DEFAULT = { maxTokens: 4096n, inPlaceSystem: false, namedToolChoice: true, reasoningSplit: false, schemaAsTool: false };

  test("a model no row names gets the defaults, on every protocol", () => {
    for (const protocol of ["anthropic_messages", "openai_chat", "openai_responses", "gemini"]) {
      expect(modelProfile(protocol, "some-model")).toEqual(DEFAULT);
    }
  });

  test("an Anthropic model's rows combine, and rows of one protocol leave the others alone", () => {
    expect(modelProfile("anthropic_messages", "claude-opus-5-5")).toEqual({
      ...DEFAULT,
      maxTokens: 128000n,
      inPlaceSystem: true,
      namedToolChoice: false,
    });
    expect(modelProfile("anthropic_messages", "claude-opus-5")).toEqual({ ...DEFAULT, maxTokens: 128000n, inPlaceSystem: true });
    expect(modelProfile("openai_chat", "claude-opus-5-5")).toEqual(DEFAULT);
    expect(modelProfile("anthropic_messages", "MiniMax-M2.7")).toEqual(DEFAULT);
  });

  test("max_tokens goes by exact id; the other Anthropic rows by prefix", () => {
    expect(modelProfile("anthropic_messages", "claude-sonnet-4-5-20250929").maxTokens).toBe(64000n);
    expect(modelProfile("anthropic_messages", "claude-sonnet-4-5-20990101").maxTokens).toBe(4096n);
    expect(modelProfile("anthropic_messages", "claude-opus-5-5-20260101").namedToolChoice).toBe(false);
    expect(modelProfile("anthropic_messages", "claude-mythos-5-20260101").inPlaceSystem).toBe(true);
  });

  test("DeepSeek's thinking models and MiniMax M2.x refuse a named tool choice; MiniMax splits reasoning", () => {
    for (const model of ["deepseek-reasoner", "deepseek-v4-pro", "deepseek-v5-pro"]) {
      expect(modelProfile("openai_chat", model)).toEqual({ ...DEFAULT, namedToolChoice: false });
    }
    for (const model of ["MiniMax-M2", "MiniMax-M2.7", "MiniMax-M3"]) {
      expect(modelProfile("openai_chat", model)).toEqual({ ...DEFAULT, namedToolChoice: false, reasoningSplit: true });
    }
    expect(modelProfile("openai_chat", "deepseek-chat")).toEqual(DEFAULT);
    expect(modelProfile("openai_chat", "MiniMax-Text-01")).toEqual(DEFAULT);
  });

  test.each(["gpt-3.5-turbo", "gpt-35-turbo", "gpt-4", "gpt-4-turbo", "gpt-4-0613"])("%s sends its schema as a tool", (model) => {
    expect(modelProfile("openai_chat", model).schemaAsTool).toBe(true);
  });

  test.each(["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-40", "gpt-5", "gpt-5.5", "o3", "my-deploy", ""])(
    "%p sends its schema as a JSON-schema format",
    (model) => {
      expect(modelProfile("openai_chat", model).schemaAsTool).toBe(false);
    },
  );

  test("a profile is frozen", () => {
    expect(Object.isFrozen(modelProfile("anthropic_messages", "claude-opus-5-5"))).toBe(true);
    expect(Object.isFrozen(modelProfile("openai_chat", "gpt-4o"))).toBe(true);
  });

  // A catalog model missing from the table would fall back to 4096 and cut long outputs.
  const claudes = getKnownModels()["anthropic"] ?? [];
  test("every Anthropic catalog model has its own max_tokens", () => {
    expect(claudes.length).toBeGreaterThan(0);
    for (const model of claudes) expect(modelProfile("anthropic_messages", model).maxTokens, model).not.toBe(4096n);
  });
});

describe("supportsNativeStructured", () => {
  test("anthropic, openai and google support native", () => {
    expect(supportsNativeStructured("anthropic", "claude-sonnet-4-5")).toBe(true);
    expect(supportsNativeStructured("openai", "gpt-5.5")).toBe(true);
    expect(supportsNativeStructured("google", "gemini-3-pro")).toBe(true);
  });

  test("the no-native sentinel returns false", () => {
    expect(supportsNativeStructured("vllm", "no-structured-sentinel")).toBe(false);
  });

  test("unknown defaults to native", () => {
    expect(supportsNativeStructured("some-future-provider", "some-model")).toBe(true);
  });

  test("provider is compared case-insensitively", () => {
    expect(supportsNativeStructured("AZURE", "no-structured-sentinel")).toBe(true);
  });
});
