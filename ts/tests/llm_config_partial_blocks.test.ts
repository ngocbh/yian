/** Mirrors `tests/test_llm_config_partial_blocks.py`: all tests. */

import { describe, expect, test } from "bun:test";

import {
  BlockProblem,
  blockFieldAnnotation,
  locatedBlockProblems,
  partialBlockProblems,
} from "../src/llm_clients/provider_blocks.ts";

const SENTINEL = "sk-sentinel-5f0c2a9e";

/** Stands in for a value (`${...}`) in a block, resolved only later. */
class Value {}

const isValue = (leaf: unknown): boolean => leaf instanceof Value;

const problems = (block: string, tree: unknown): string[] => partialBlockProblems(block, tree, isValue);

describe("partialBlockProblems", () => {
  test("a block of values and valid literals has no problems", () => {
    const tree = {
      max_tokens: new Value(),
      temperature: 0.5,
      thinking: { type: "enabled", budget_tokens: 2048n },
    };

    expect(problems("anthropic", tree)).toEqual([]);
  });

  test("unknown and reserved key paths are reported beside values", () => {
    const found = problems("anthropic", { temprature: new Value(), stream: true, max_tokens: new Value() });

    expect(found).toHaveLength(2);
    expect(found[0]).toStartWith(
      "`anthropic.temprature` is not a supported field of the Anthropic Messages API request;",
    );
    expect(found[1]).toBe("`anthropic.stream` is built by yian (streaming); remove it");
  });

  test("a reserved nested path is reported", () => {
    expect(problems("anthropic", { output_config: { format: new Value(), effort: "high" } })).toEqual([
      "`anthropic.output_config.format` is built by yian (structured output); remove it",
    ]);
  });

  test("an unknown nested key is reported beside a value", () => {
    const found = problems("google", { generationConfig: { topK: new Value(), topk: 3n } });

    expect(found).toHaveLength(1);
    expect(found[0]).toStartWith("`google.generationConfig.topk` is not a supported field");
  });

  test("a value leaf is opaque even where a block is expected", () => {
    expect(problems("anthropic", { thinking: new Value(), metadata: new Value() })).toEqual([]);
  });

  test("a literal leaf is checked against its field", () => {
    const tree = { thinking: { type: "enabled", budget_tokens: 10n }, temperature: new Value() };

    expect(problems("anthropic", tree)).toEqual([
      "`anthropic.thinking.budget_tokens`: Input should be greater than or equal to 1024",
    ]);
  });

  test("literal leaves are checked by strict type and literal", () => {
    const tree = { max_tokens: "8000", service_tier: "fastest", top_k: new Value() };

    expect(problems("anthropic", tree)).toEqual([
      "`anthropic.max_tokens`: Input should be a valid integer",
      "`anthropic.service_tier`: Input should be 'auto' or 'standard_only'",
    ]);
  });

  test("a required nested field given as a value counts as present", () => {
    expect(problems("anthropic", { thinking: { type: "enabled", budget_tokens: new Value() } })).toEqual([]);
  });

  test("a missing required nested field is reported", () => {
    expect(problems("anthropic", { thinking: { type: "enabled", display: new Value() } })).toEqual([
      "`anthropic.thinking.budget_tokens` is required",
    ]);
  });

  test("a discriminator given as a value skips the nested keys", () => {
    const tree = { thinking: { type: new Value(), budget_tokens: 10n, bogus: 1n } };

    expect(problems("anthropic", tree)).toEqual([]);
  });

  test("an unknown literal discriminator beside a value is reported", () => {
    expect(problems("anthropic", { thinking: { type: "sometimes", display: new Value() } })).toEqual([
      "`anthropic.thinking`: `type` must be one of: adaptive, between_tools, disabled, enabled",
    ]);
  });

  test("list items are walked for key paths", () => {
    const tree = {
      safetySettings: [
        { category: new Value(), threshold: "OFF" },
        { category: new Value(), treshold: "OFF" },
      ],
    };

    const found = problems("google", tree);

    expect(found).toHaveLength(2);
    expect(found[0]).toStartWith("`google.safetySettings[1].treshold` is not a supported field");
    expect(found[1]).toBe("`google.safetySettings[1].threshold` is required");
  });

  test("located problems give each key path as parts", () => {
    const tree = {
      safetySettings: [{ category: new Value(), treshold: "OFF" }],
      generationConfig: { topK: "3" },
    };

    const located = locatedBlockProblems("google", tree, isValue);

    expect(located.map((problem) => problem.keys)).toEqual([
      ["safetySettings", 0, "treshold"],
      ["safetySettings", 0, "threshold"],
      ["generationConfig", "topK"],
    ]);
    expect(located.map((problem) => problem.message)).toEqual(problems("google", tree));
    expect(locatedBlockProblems("anthropic", [], isValue)).toEqual([
      new BlockProblem([], "`anthropic` must be a mapping"),
    ]);
  });

  test("a literal list is checked whole", () => {
    expect(problems("anthropic", { stop_sequences: ["a", 1n], max_tokens: new Value() })).toEqual([
      "`anthropic.stop_sequences[1]`: Input should be a valid string",
    ]);
  });

  test("messages never echo values", () => {
    const tree = {
      service_tier: SENTINEL,
      metadata: { user_id: SENTINEL.repeat(40) },
      max_tokens: SENTINEL,
      thinking: { type: SENTINEL, display: new Value() },
      temperature: new Value(),
    };

    const found = problems("anthropic", tree);

    expect(found).toHaveLength(4);
    expect(found.some((problem) => problem.includes(SENTINEL))).toBe(false);
  });

  test("a block that is not a mapping is reported", () => {
    expect(problems("anthropic", [1n])).toEqual(["`anthropic` must be a mapping"]);
  });

  test("a literal union leaf reports key paths without union branch names", () => {
    expect(problems("deepseek", { stop: ["a", 5n], max_tokens: new Value() })).toEqual([
      "`deepseek.stop`: Input should be a valid string",
      "`deepseek.stop[1]`: Input should be a valid string",
    ]);
  });

  test("literal items of a mapping holding a value are checked", () => {
    expect(problems("openai", { metadata: { a: 5n, b: new Value() } })).toEqual([
      "`openai.metadata.a`: Input should be a valid string",
    ]);
  });
});

describe("blockFieldAnnotation", () => {
  test("the annotation of a bounded integer field", () => {
    const tree = { thinking: { type: "enabled", budget_tokens: new Value() } };

    expect(blockFieldAnnotation("anthropic", ["thinking", "budget_tokens"], tree)).toEqual({
      base: "int",
      literals: [],
      ge: 1024n,
      le: null,
    });
  });

  test("the annotations of simple fields", () => {
    expect(blockFieldAnnotation("anthropic", ["temperature"], {})).toEqual({
      base: "float",
      literals: [],
      ge: 0,
      le: 1,
    });
    expect(blockFieldAnnotation("anthropic", ["service_tier"], {})).toEqual({
      base: "str",
      literals: ["auto", "standard_only"],
      ge: null,
      le: null,
    });
    expect(blockFieldAnnotation("google", ["store"], {})).toEqual({ base: "bool", literals: [], ge: null, le: null });
    expect(blockFieldAnnotation("anthropic", ["inference_geo"], {})).toEqual({
      base: "str",
      literals: [],
      ge: null,
      le: null,
    });
  });

  test.each([
    ["openai", ["metadata"], {}],
    ["anthropic", ["thinking"], {}],
    ["anthropic", ["stop_sequences"], {}],
    ["ollama", ["think"], {}],
    ["anthropic", ["thinking", "budget_tokens"], { thinking: { type: new Value() } }],
    ["anthropic", ["thinking", "budget_tokens"], {}],
    ["anthropic", ["nope"], {}],
  ] as const)("no annotation for fields checked when the value resolves: %s %p", (block, path, tree) => {
    expect(blockFieldAnnotation(block, path, tree)).toBeNull();
  });
});

describe("TS additions", () => {
  test("a top-level list without values is key-walked as Python's assignment check does", () => {
    expect(locatedBlockProblems("google", { safetySettings: [{ bogus: 1n }] }, isValue)).toEqual([
      new BlockProblem(
        ["safetySettings"],
        "`google.safetySettings`: Value error, `google.safetySettings[0].bogus` is not a supported field " +
          "of the Gemini generateContent request; allowed: category, threshold",
      ),
    ]);
  });

  test("an unknown block throws RangeError even when the tree is a value", () => {
    expect(() => locatedBlockProblems("nope", new Value(), isValue)).toThrow(RangeError);
    expect(() => partialBlockProblems("nope", new Value(), isValue)).toThrow(RangeError);
    expect(() => blockFieldAnnotation("nope", [], new Value())).toThrow(RangeError);
  });

  test("`constructor` as a nested key is reported as unknown", () => {
    expect(problems("anthropic", { thinking: { type: "enabled", budget_tokens: 2048n, constructor: 1n } })).toEqual([
      "`anthropic.thinking.constructor` is not a supported field of the Anthropic Messages API request; " +
        "allowed: budget_tokens, display, type",
    ]);
    expect(problems("anthropic", { metadata: { constructor: 5n } })).toEqual([
      "`anthropic.metadata.constructor` is not a supported field of the Anthropic Messages API request; " +
        "allowed: user_id",
    ]);
    expect(blockFieldAnnotation("anthropic", ["constructor"], {})).toBeNull();
  });

  test("null leaves an optional field unset; a required field given null is checked", () => {
    expect(problems("anthropic", { temperature: null, top_p: new Value() })).toEqual([]);
    expect(problems("anthropic", { thinking: { type: "enabled", budget_tokens: null } })).toEqual([
      "`anthropic.thinking.budget_tokens`: Input should be a valid integer",
    ]);
    expect(problems("openai", { metadata: { a: null, b: new Value() } })).toEqual([
      "`openai.metadata.a`: Input should be a valid string",
    ]);
  });

  test("a problem and its keys are frozen", () => {
    const [problem] = locatedBlockProblems("anthropic", { stream: true }, isValue);

    expect(Object.isFrozen(problem)).toBe(true);
    expect(Object.isFrozen(problem!.keys)).toBe(true);
  });
});
