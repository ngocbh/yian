/**
 * Mirrors `tests/test_provider_blocks.py`. Not ported: the SDK-signature tests
 * (`test_responses_fields_and_reserved_keys_are_keywords_of_responses_create`,
 * `test_responses_nested_fields_are_keys_of_the_sdk_params`,
 * `test_ollama_block_fields_are_keywords_of_client_chat`; TS has no SDKs) and the
 * `merge_request_body` tests, which live with `mergeRequestBody` in `tests/llm/body.test.ts`
 * (TS has no `top_level`/`extra_body`).
 */

import { describe, expect, test } from "bun:test";

import { compareCodeUnits } from "../src/diagnostics.ts";
import { FileRef } from "../src/typesys/index.ts";
import {
  AnthropicMessagesBlock,
  type BlockSchema,
  checkBlock,
  checkObject,
  type Field,
  GeminiGenerateContentBlock,
  type Issue,
  type ObjectSchema,
  OpenAIResponsesBlock,
  OpenAISubscriptionBlock,
  ValidationError,
  validateBlock,
} from "../src/llm_clients/provider_blocks.ts";

/** Return a frozen schema with `fields`, of which `required` are required. */
function schema(fields: Record<string, Field>, required: string[] = []): ObjectSchema {
  return Object.freeze({ fields: Object.freeze(fields), required: Object.freeze(required) });
}

/** Return the `{loc, type, msg}` of each issue `checkObject` finds in `data`. */
function issues(of: ObjectSchema, data: unknown): Pick<Issue, "loc" | "type" | "msg">[] {
  const result = checkObject(of, data);
  if (result.issues.length > 0) expect(result.value).toBeNull();
  return result.issues.map(({ loc, type, msg }) => ({ loc, type, msg }));
}

/** Return the value `checkObject` gives for `data`, asserting there are no issues. */
function value(of: ObjectSchema, data: unknown): unknown {
  const result = checkObject(of, data);
  expect(result.issues).toEqual([]);
  return result.value;
}

const STR: Field = { kind: "str" };
const INT: Field = { kind: "int" };

describe("value pass", () => {
  test("str takes a string and checks short, long, then pattern, in code points", () => {
    const s = schema({ v: { kind: "str", minLength: 6, maxLength: 512, pattern: "^[a-z]+$" } });
    expect(value(s, { v: "abcdefg" })).toEqual({ v: "abcdefg" });
    expect(issues(s, { v: 5n })).toEqual([{ loc: ["v"], type: "string_type", msg: "Input should be a valid string" }]);
    expect(issues(s, { v: "!".repeat(600) })).toEqual([
      { loc: ["v"], type: "string_too_long", msg: "String should have at most 512 characters" },
    ]);
    expect(issues(s, { v: "!" })).toEqual([
      { loc: ["v"], type: "string_too_short", msg: "String should have at least 6 characters" },
    ]);
    expect(issues(s, { v: "abcdef!" })).toEqual([
      { loc: ["v"], type: "string_pattern_mismatch", msg: "String should match pattern '^[a-z]+$'" },
    ]);
    const points = schema({ v: { kind: "str", minLength: 2, maxLength: 2 } });
    expect(value(points, { v: "😀😀" })).toEqual({ v: "😀😀" });
    expect(issues(points, { v: "😀" })).toEqual([
      { loc: ["v"], type: "string_too_short", msg: "String should have at least 2 characters" },
    ]);
  });

  test("int takes only a bigint", () => {
    const s = schema({ v: INT });
    expect(value(s, { v: 4096n })).toEqual({ v: 4096n });
    for (const bad of [4096, true, "5"]) {
      expect(issues(s, { v: bad })).toEqual([{ loc: ["v"], type: "int_type", msg: "Input should be a valid integer" }]);
    }
  });

  test("float takes a finite number and keeps a bigint as is", () => {
    const s = schema({ v: { kind: "float" } });
    const kept = value(s, { v: 1n }) as { v: unknown };
    expect(typeof kept.v).toBe("bigint");
    expect(kept.v).toBe(1n);
    expect(value(s, { v: 0.5 })).toEqual({ v: 0.5 });
    for (const bad of [true, "1", 10n ** 400n]) {
      expect(issues(s, { v: bad })).toEqual([
        { loc: ["v"], type: "float_type", msg: "Input should be a valid number" },
      ]);
    }
    for (const bad of [Infinity, -Infinity, NaN]) {
      expect(issues(s, { v: bad })).toEqual([
        { loc: ["v"], type: "finite_number", msg: "Input should be a finite number" },
      ]);
    }
  });

  test("int and float bounds print the bound without an n", () => {
    const s = schema({
      ge: { kind: "int", ge: 1024n },
      gt: { kind: "float", gt: 0 },
      le: { kind: "int", le: 20n },
      lt: { kind: "float", lt: 2 },
      frac: { kind: "float", ge: 0.01, le: 1 },
    });
    expect(issues(s, { ge: 5n, gt: 0, le: 21n, lt: 2n, frac: 0.001 })).toEqual([
      { loc: ["ge"], type: "greater_than_equal", msg: "Input should be greater than or equal to 1024" },
      { loc: ["gt"], type: "greater_than", msg: "Input should be greater than 0" },
      { loc: ["le"], type: "less_than_equal", msg: "Input should be less than or equal to 20" },
      { loc: ["lt"], type: "less_than", msg: "Input should be less than 2" },
      { loc: ["frac"], type: "greater_than_equal", msg: "Input should be greater than or equal to 0.01" },
    ]);
    expect(value(s, { ge: 1024n, gt: 1n, le: 20n, lt: 1.5, frac: 1n })).toEqual({
      ge: 1024n,
      gt: 1n,
      le: 20n,
      lt: 1.5,
      frac: 1n,
    });
  });

  test("bool takes only a boolean", () => {
    const s = schema({ v: { kind: "bool" } });
    expect(value(s, { v: false })).toEqual({ v: false });
    for (const bad of ["true", 1n]) {
      expect(issues(s, { v: bad })).toEqual([
        { loc: ["v"], type: "bool_type", msg: "Input should be a valid boolean" },
      ]);
    }
  });

  test("literal takes one of its strings and lists them in the message", () => {
    const s = schema({
      three: { kind: "literal", values: ["a", "b", "c"] },
      two: { kind: "literal", values: ["a", "b"] },
      one: { kind: "literal", values: ["a"] },
    });
    expect(value(s, { three: "c", two: "a", one: "a" })).toEqual({ three: "c", two: "a", one: "a" });
    expect(issues(s, { three: "d", two: 1n, one: "A" })).toEqual([
      { loc: ["three"], type: "literal_error", msg: "Input should be 'a', 'b' or 'c'" },
      { loc: ["two"], type: "literal_error", msg: "Input should be 'a' or 'b'" },
      { loc: ["one"], type: "literal_error", msg: "Input should be 'a'" },
    ]);
  });

  test("list checks each item, or only its length when too long", () => {
    const s = schema({ v: { kind: "list", item: STR, maxLength: 4 } });
    expect(value(s, { v: ["a", "b"] })).toEqual({ v: ["a", "b"] });
    expect(issues(s, { v: "a" })).toEqual([{ loc: ["v"], type: "list_type", msg: "Input should be a valid list" }]);
    expect(issues(s, { v: ["a", 5n, undefined] })).toEqual([
      { loc: ["v", 1], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["v", 2], type: "string_type", msg: "Input should be a valid string" },
    ]);
    expect(issues(s, { v: [1n, 2n, 3n, 4n, 5n] })).toEqual([
      { loc: ["v"], type: "too_long", msg: "List should have at most 4 items after validation, not 5" },
    ]);
  });

  test("map checks each key then value, and its length only when they pass", () => {
    const s = schema({
      v: { kind: "map", key: { kind: "str", maxLength: 2 }, value: { kind: "str", maxLength: 3 }, maxLength: 16 },
    });
    expect(issues(s, { v: 5n })).toEqual([
      { loc: ["v"], type: "dict_type", msg: "Input should be a valid dictionary" },
    ]);
    expect(issues(s, { v: { abc: "defg", b: 1n } })).toEqual([
      { loc: ["v", "abc", "[key]"], type: "string_too_long", msg: "String should have at most 2 characters" },
      { loc: ["v", "abc"], type: "string_too_long", msg: "String should have at most 3 characters" },
      { loc: ["v", "b"], type: "string_type", msg: "Input should be a valid string" },
    ]);
    const keys = Array.from({ length: 17 }, (_, i) => `k${i.toString(36)}`);
    const bad = issues(s, { v: Object.fromEntries(keys.map((k) => [k, 1n])) });
    expect(bad).toEqual(
      keys.map((k) => ({ loc: ["v", k], type: "string_type", msg: "Input should be a valid string" })),
    );
    expect(issues(s, { v: Object.fromEntries(keys.map((k) => [k, "x"])) })).toEqual([
      { loc: ["v"], type: "too_long", msg: "Dictionary should have at most 16 items after validation, not 17" },
    ]);
  });

  test("map entries holding undefined are skipped and not counted", () => {
    const s = schema({ v: { kind: "map", key: STR, value: STR, maxLength: 2 } });
    const result = value(s, { v: { a: "x", b: undefined, c: "y" } }) as { v: object };
    expect(Object.entries(result.v)).toEqual([
      ["a", "x"],
      ["c", "y"],
    ]);
  });

  test("a __proto__ key in a map is an own key of the result", () => {
    const s = schema({ v: { kind: "map", key: STR, value: STR } });
    const result = value(s, { v: Object.fromEntries([["__proto__", "x"]]) }) as { v: object };
    expect(Object.hasOwn(result.v, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(result.v)).toBe(Object.prototype);
    expect(Object.entries(result.v)).toEqual([["__proto__", "x"]]);
  });

  test("object checks fields in declaration order, then extras in written order", () => {
    const inner = schema({ a: STR, b: INT }, ["b"]);
    const s = schema({ first: STR, nested: { kind: "object", schema: inner }, need: INT }, ["need"]);
    expect(issues(s, 5n)).toEqual([{ loc: [], type: "model_type", msg: "Input should be a valid dictionary" }]);
    expect(issues(s, ["a"])).toEqual([{ loc: [], type: "model_type", msg: "Input should be a valid dictionary" }]);
    expect(issues(s, { zz: 1n, nested: { a: 1n, x: 1n }, toString: 1n, first: 1n })).toEqual([
      { loc: ["first"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["nested", "a"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["nested", "b"], type: "missing", msg: "Field required" },
      { loc: ["nested", "x"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
      { loc: ["need"], type: "missing", msg: "Field required" },
      { loc: ["zz"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
      { loc: ["toString"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
    ]);
    expect(issues(s, { nested: "x", need: 1n })).toEqual([
      { loc: ["nested"], type: "model_type", msg: "Input should be a valid dictionary" },
    ]);
    // A required field given null is checked against its type; an optional one is unset.
    expect(issues(s, { first: null, need: null })).toEqual([
      { loc: ["need"], type: "int_type", msg: "Input should be a valid integer" },
    ]);
    expect(value(s, { first: null, nested: { a: null, b: 1n }, need: 2n })).toEqual({ nested: { b: 1n }, need: 2n });
  });

  test("undefined reads as absent", () => {
    const s = schema({ a: STR, need: INT }, ["need"]);
    expect(issues(s, { a: undefined, need: undefined, extra: undefined })).toEqual([
      { loc: ["need"], type: "missing", msg: "Field required" },
    ]);
    const result = value(s, { a: undefined, need: 1n, extra: undefined }) as object;
    expect(Object.keys(result)).toEqual(["need"]);
  });

  test("union takes the first member that passes and labels each member's issues", () => {
    const s = schema({
      stop: {
        kind: "union",
        members: [STR, { kind: "list", item: STR, maxLength: 4 }, { kind: "list", item: INT }],
      },
      keep: { kind: "union", members: [{ kind: "float" }, STR] },
      think: { kind: "union", members: [{ kind: "bool" }, { kind: "literal", values: ["low", "high"] }] },
    });
    const good = { stop: [1n, 2n], keep: "5m", think: "low" };
    expect(value(s, good)).toEqual(good);
    expect(issues(s, { stop: ["a", 5n], keep: true, think: 5n })).toEqual([
      { loc: ["stop", "str"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["stop", "list[str]", 1], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["stop", "list[int]", 0], type: "int_type", msg: "Input should be a valid integer" },
      { loc: ["keep", "float"], type: "float_type", msg: "Input should be a valid number" },
      { loc: ["keep", "str"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["think", "bool"], type: "bool_type", msg: "Input should be a valid boolean" },
      { loc: ["think", "literal['low','high']"], type: "literal_error", msg: "Input should be 'low' or 'high'" },
    ]);
  });

  test("tagged picks the variant its type names, never echoing an unknown tag", () => {
    const enabled = schema({ type: { kind: "literal", values: ["enabled"] }, budget: { kind: "int", ge: 1024n } }, [
      "type",
      "budget",
    ]);
    const disabled = schema({ type: { kind: "literal", values: ["disabled"] } }, ["type"]);
    const s = schema({ thinking: { kind: "tagged", variants: Object.freeze({ enabled, disabled }) } });
    const unknown = { loc: ["thinking"], type: "unknown_type", msg: "`type` must be one of: disabled, enabled" };
    const bads = [5n, ["enabled"], {}, { type: "other" }, { type: "constructor" }, { type: 1n }, { type: undefined }];
    for (const bad of bads) {
      expect(issues(s, { thinking: bad })).toEqual([unknown]);
    }
    expect(issues(s, { thinking: { type: "enabled" } })).toEqual([
      { loc: ["thinking", "enabled", "budget"], type: "missing", msg: "Field required" },
    ]);
    expect(issues(s, { thinking: { type: "enabled", budget: 5n } })).toEqual([
      {
        loc: ["thinking", "enabled", "budget"],
        type: "greater_than_equal",
        msg: "Input should be greater than or equal to 1024",
      },
    ]);
    expect(value(s, { thinking: { type: "disabled" } })).toEqual({ thinking: { type: "disabled" } });
  });

  test("json takes JSON values at any depth and refuses the rest at the inner location", () => {
    const s = schema({ v: { kind: "map", key: STR, value: { kind: "json" } } });
    const ok = { a: null, b: [true, "x", 1n, 0.5, { c: [] }], d: { e: null } };
    expect(value(s, { v: ok })).toEqual({ v: ok });
    expect(issues(s, { v: { a: [Infinity] } })).toEqual([
      { loc: ["v", "a", 0], type: "finite_number", msg: "Input should be a finite number" },
    ]);
    const file = new FileRef({ uri: "file:///x.txt" });
    expect(issues(s, { v: { a: { b: NaN }, c: [undefined], d: file, e: () => 1, f: new Map() } })).toEqual([
      { loc: ["v", "a", "b"], type: "finite_number", msg: "Input should be a finite number" },
      { loc: ["v", "c", 0], type: "json_type", msg: "Input should be a JSON value" },
      { loc: ["v", "d"], type: "json_type", msg: "Input should be a JSON value" },
      { loc: ["v", "e"], type: "json_type", msg: "Input should be a JSON value" },
      { loc: ["v", "f"], type: "json_type", msg: "Input should be a JSON value" },
    ]);
  });

  test("json record entries holding undefined are skipped and not counted", () => {
    const s = schema({ v: { kind: "map", key: STR, value: { kind: "json" }, maxLength: 1 } });
    const result = value(s, { v: { a: { b: undefined, c: 1n }, d: undefined } }) as { v: { a: object } };
    expect(result).toEqual({ v: { a: { c: 1n } } });
    expect(Object.keys(result.v.a)).toEqual(["c"]);
  });

  test("the result is a fresh copy and the input is left as it was", () => {
    const s = schema({
      nested: { kind: "object", schema: schema({ a: STR, b: STR }) },
      list: { kind: "list", item: STR },
      map: { kind: "map", key: STR, value: { kind: "json" } },
    });
    const input = { nested: { a: "x", b: null }, list: ["y"], map: { k: { deep: [1n] } } };
    const before = structuredClone(input);
    const result = value(s, input) as { nested: object; list: string[]; map: { k: { deep: bigint[] } } };
    expect(input).toEqual(before);
    expect(result).toStrictEqual({ nested: { a: "x" }, list: ["y"], map: { k: { deep: [1n] } } });
    expect(result.nested).not.toBe(input.nested);
    expect(result.list).not.toBe(input.list);
    expect(result.map).not.toBe(input.map);
    expect(result.map.k).not.toBe(input.map.k);
    expect(result.map.k.deep).not.toBe(input.map.k.deep);
  });
});

describe("ValidationError", () => {
  test("the message has a count line, dotted locations and two-space messages", () => {
    const given: Issue[] = [
      { loc: ["m", "abc", "[key]"], type: "string_too_long", msg: "String should have at most 2 characters" },
      { loc: ["stop", "list[str]", 1], type: "string_type", msg: "Input should be a valid string" },
      { loc: [], type: "value_error", msg: "Value error, bad", error: "bad" },
    ];
    const error = new ValidationError("SomeBlock", given);
    expect(error.message).toBe(
      "3 validation errors for SomeBlock\n" +
        "m.abc.[key]\n  String should have at most 2 characters\n" +
        "stop.list[str].1\n  Input should be a valid string\n" +
        "  Value error, bad",
    );
    expect(error.name).toBe("ValidationError");
    expect(error).toBeInstanceOf(RangeError);
    expect(error.title).toBe("SomeBlock");
    expect(Object.isFrozen(error.issues)).toBe(true);
    expect(error.issues).toEqual(given);
    error.issues.forEach((issue, i) => {
      expect(issue).not.toBe(given[i]);
      expect(issue.loc).not.toBe(given[i]!.loc);
      expect(Object.isFrozen(issue)).toBe(true);
      expect(Object.isFrozen(issue.loc)).toBe(true);
    });
    expect(error.issues[2]!.error).toBe("bad");
    const one = new ValidationError("B", [{ loc: ["a"], type: "missing", msg: "Field required" }]);
    expect(one.message).toBe("1 validation error for B\na\n  Field required");
  });
});

const SENTINEL = "sk-sentinel-7d2e91b4";

const RESPONSES_SAMPLE = {
  access_programs: { cyber: "standard" },
  max_output_tokens: 2048n,
  max_tool_calls: 4n,
  metadata: { team: "research" },
  moderation: {
    model: "omni-moderation-latest",
    policy: { input: { mode: "block" }, output: { mode: "score" } },
  },
  prompt_cache_key: "flow-a",
  prompt_cache_options: { mode: "explicit", ttl: "30m" },
  prompt_cache_retention: "24h",
  reasoning: { context: "all_turns", effort: "high", mode: "pro", summary: "auto" },
  safety_identifier: "user-hash",
  service_tier: "flex",
  temperature: 0.2,
  text: { verbosity: "low" },
  top_logprobs: 5n,
  top_p: 0.9,
  truncation: "auto",
  user: "someone",
};

const ANTHROPIC_SAMPLE = {
  cache_control: { type: "ephemeral", ttl: "1h" },
  inference_geo: "us",
  max_tokens: 8192n,
  metadata: { user_id: "user-hash" },
  output_config: { effort: "xhigh" },
  service_tier: "standard_only",
  stop_sequences: ["END"],
  temperature: 1n,
  thinking: { type: "enabled", budget_tokens: 2048n, display: "summarized" },
  top_k: 40n,
  top_p: 0.99,
};

const GEMINI_SAMPLE = {
  generationConfig: {
    stopSequences: ["END"],
    maxOutputTokens: 4096n,
    temperature: 0.7,
    topP: 0.95,
    topK: 40n,
    seed: 7n,
    presencePenalty: 0.5,
    frequencyPenalty: -0.5,
    responseLogprobs: true,
    logprobs: 5n,
    enableEnhancedCivicAnswers: false,
    thinkingConfig: { includeThoughts: true, thinkingLevel: "HIGH" },
    mediaResolution: "MEDIA_RESOLUTION_LOW",
  },
  safetySettings: [
    { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
    { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "OFF" },
  ],
  serviceTier: "flex",
  store: false,
};

/** Return the message of the `ValidationError` that `validateBlock` throws for `data`. */
function error(of: BlockSchema, data: unknown): string {
  try {
    validateBlock(of, data);
  } catch (caught) {
    expect(caught).toBeInstanceOf(ValidationError);
    expect((caught as ValidationError).title).toBe(of.title);
    return (caught as ValidationError).message;
  }
  throw new Error("expected a ValidationError");
}

/** Return `value` nested under the keys of the dotted `path`. */
function nest(path: string, value: unknown): unknown {
  return path
    .split(".")
    .reverse()
    .reduce((inner, key) => Object.fromEntries([[key, inner]]), value);
}

describe("OpenAI Responses block", () => {
  test("dump omits unset fields at every level", () => {
    const expected = { reasoning: { effort: "low" }, text: {}, top_p: 0.5 };
    expect(validateBlock(OpenAIResponsesBlock, { reasoning: { effort: "low" }, text: {}, top_p: 0.5 })).toEqual(
      expected,
    );
    expect(validateBlock(OpenAIResponsesBlock, {})).toEqual({});
  });

  test("explicit null reads as unset", () => {
    const data = { temperature: null, reasoning: { effort: null, summary: "concise" } };
    expect(validateBlock(OpenAIResponsesBlock, data)).toEqual({ reasoning: { summary: "concise" } });
  });

  test("top-level typo names the block path and the allowed keys", () => {
    const message = error(OpenAIResponsesBlock, { temprature: 0.3 });
    expect(message).toContain("`openai.temprature` is not a supported field of the OpenAI Responses API request");
    expect(message).toContain("allowed: access_programs, max_output_tokens, max_tool_calls, metadata,");
    expect(message).toContain("temperature");
  });

  test("nested typo names the full path and the nested allowed keys", () => {
    let message = error(OpenAIResponsesBlock, { reasoning: { efort: "high" } });
    expect(message).toContain("`openai.reasoning.efort` is not a supported field of the OpenAI Responses API request");
    expect(message).toContain("allowed: context, effort, mode, summary");

    message = error(OpenAIResponsesBlock, { text: { verbose: "low" } });
    expect(message).toContain("`openai.text.verbose`");
    expect(message).toContain("allowed: verbosity");
  });

  test("every problem is reported", () => {
    const message = error(OpenAIResponsesBlock, { temprature: 1n, store: false });
    expect(message).toContain("`openai.temprature`");
    expect(message).toContain("`openai.store`");
  });

  test("key problems are one value error at the block, in written order", () => {
    const joined =
      "`openai.temprature` is not a supported field of the OpenAI Responses API request; allowed: " +
      "access_programs, max_output_tokens, max_tool_calls, metadata, moderation, prompt_cache_key, " +
      "prompt_cache_options, prompt_cache_retention, reasoning, safety_identifier, service_tier, " +
      "temperature, text, top_logprobs, top_p, truncation, user; " +
      "`openai.store` is built by yian (conversation state); remove it";
    const result = checkBlock(OpenAIResponsesBlock, { temprature: 1n, store: false, skipped: undefined });
    expect(result.value).toBeNull();
    expect(result.issues).toEqual([{ loc: [], type: "value_error", msg: `Value error, ${joined}`, error: joined }]);
    expect(error(OpenAIResponsesBlock, { temprature: 1n, store: false })).toBe(
      `1 validation error for OpenAIResponsesBlock\n  Value error, ${joined}`,
    );
  });

  test("a non-record is one model_type issue at the block", () => {
    for (const bad of [5n, "x", null, ["a"]]) {
      expect(checkBlock(OpenAIResponsesBlock, bad)).toEqual({
        value: null,
        issues: [{ loc: [], type: "model_type", msg: "Input should be a valid dictionary" }],
      });
    }
  });

  const badValues: unknown[] = [
    { max_output_tokens: "x" },
    { max_output_tokens: "5" },
    { max_output_tokens: 0n },
    { max_output_tokens: true },
    { reasoning: { effort: "hgih" } },
    { reasoning: { summary: "brief" } },
    { reasoning: { context: "every_turn" } },
    { reasoning: { mode: "turbo" } },
    { reasoning: "high" },
    { text: { verbosity: "loud" } },
    { temperature: 3n },
    { temperature: -0.1 },
    { top_p: 1.5 },
    { top_logprobs: 21n },
    { truncation: "on" },
    { service_tier: "turbo" },
    { metadata: { k: 1n } },
    { metadata: Object.fromEntries(Array.from({ length: 17 }, (_, i) => [`k${i}`, "v"])) },
    { access_programs: { cyber: "daybreak_green" } },
    { moderation: { policy: { input: { mode: "block" } } } },
    { moderation: { model: "omni-moderation-latest", policy: { input: {} } } },
    { moderation: { model: "omni-moderation-latest", policy: { output: { mode: "warn" } } } },
    { prompt_cache_options: { ttl: "1h" } },
    { prompt_cache_options: { mode: "auto" } },
  ];
  test.each(badValues.map((data) => [data]))("bad value is rejected: %p", (data) => {
    expect(() => validateBlock(OpenAIResponsesBlock, data)).toThrow(ValidationError);
  });

  const reservedRows: [unknown, string][] = [
    [{ input: "hi" }, "input"],
    [{ instructions: "be terse" }, "instructions"],
    [{ tools: [] }, "tools"],
    [{ tool_choice: "auto" }, "tool_choice"],
    [{ parallel_tool_calls: false }, "parallel_tool_calls"],
    [{ model: "gpt-5" }, "model"],
    [{ text: { format: { type: "text" } } }, "text.format"],
    [{ store: false }, "store"],
    [{ include: [] }, "include"],
    [{ previous_response_id: "resp_1" }, "previous_response_id"],
    [{ conversation: "conv_1" }, "conversation"],
    [{ prompt: { id: "pmpt_1" } }, "prompt"],
    [{ background: true }, "background"],
    [{ stream: true }, "stream"],
    [{ stream_options: {} }, "stream_options"],
    [{ context_management: [{ type: "compaction" }] }, "context_management"],
  ];
  test.each(reservedRows)("reserved path is built by yian: %p", (data, path) => {
    const message = error(OpenAIResponsesBlock, data);
    expect(message).toContain(`\`openai.${path}\` is built by yian (`);
    expect(message).toContain("remove it");
  });

  test("a reserved key is refused whatever its value, null included", () => {
    expect(error(OpenAIResponsesBlock, { stream: null })).toContain("`openai.stream` is built by yian (streaming)");
  });

  test("reserved nested path keeps its reason", () => {
    const message = error(OpenAIResponsesBlock, { text: { format: { type: "json_object" } } });
    expect(message).toContain("`openai.text.format` is built by yian (structured output); remove it");
  });

  test.each([[NaN], [Infinity], [-Infinity]])("non-finite numbers are rejected: %p", (value) => {
    expect(() => validateBlock(OpenAIResponsesBlock, { top_p: value })).toThrow(ValidationError);
  });

  test.each([["comparison_response_id"], ["prewarm"]])(
    "per-request prompt cache options are not block fields: %s",
    (key) => {
      const message = error(OpenAIResponsesBlock, { prompt_cache_options: { [key]: true } });
      expect(message).toContain(`\`openai.prompt_cache_options.${key}\` is not a supported field of the`);
      expect(message).toContain("allowed: mode, ttl");
    },
  );

  test("valid sample dumps unchanged", () => {
    expect(validateBlock(OpenAIResponsesBlock, RESPONSES_SAMPLE)).toEqual(RESPONSES_SAMPLE);
  });

  test("fields are in declaration order: the shared fields, then max_output_tokens and temperature", () => {
    expect(Object.keys(OpenAIResponsesBlock.fields)).toEqual([
      ...Object.keys(OpenAISubscriptionBlock.fields),
      "max_output_tokens",
      "temperature",
    ]);
  });

  test("the paths the Responses adapter builds are reserved", () => {
    const paths = [
      "store",
      "stream",
      "stream_options",
      "include",
      "instructions",
      "previous_response_id",
      "tools",
      "tool_choice",
      "parallel_tool_calls",
      "text.format",
    ];
    for (const path of paths) {
      expect(Object.hasOwn(OpenAIResponsesBlock.reserved, path)).toBe(true);
      expect(error(OpenAIResponsesBlock, nest(path, "x"))).toContain(`\`openai.${path}\` is built by yian (`);
    }
  });
});

describe("OpenAI subscription block", () => {
  test("rejects fields the endpoint refuses", () => {
    for (const [key, value] of [
      ["temperature", 0.5],
      ["max_output_tokens", 100n],
    ] as const) {
      const message = error(OpenAISubscriptionBlock, { [key]: value });
      expect(message).toContain(
        `\`openai.${key}\` is not accepted by the ChatGPT subscription endpoint ` +
          "(a subset of the Responses API); allowed:",
      );
      expect(message.split("allowed:")[1]).not.toContain("temperature,");
    }
  });

  test("keeps the rest of the Responses shape", () => {
    const { temperature: _t, max_output_tokens: _m, ...sample } = RESPONSES_SAMPLE;
    expect(validateBlock(OpenAISubscriptionBlock, sample)).toEqual(sample);
    const subscription = new Set(Object.keys(OpenAISubscriptionBlock.fields));
    expect(Object.keys(OpenAIResponsesBlock.fields).filter((name) => !subscription.has(name))).toEqual([
      "max_output_tokens",
      "temperature",
    ]);
    expect(error(OpenAISubscriptionBlock, { store: true })).toContain("`openai.store` is built by yian");
  });
});

describe("Anthropic Messages block", () => {
  test("valid sample dumps unchanged", () => {
    expect(validateBlock(AnthropicMessagesBlock, ANTHROPIC_SAMPLE)).toEqual(ANTHROPIC_SAMPLE);
  });

  const accepted: unknown[] = [
    { output_config: { effort: "high" } },
    { thinking: { type: "enabled", budget_tokens: 2048n } },
    { thinking: { type: "adaptive", display: "omitted" } },
    { thinking: { type: "disabled" } },
    { thinking: { type: "between_tools" } },
    { cache_control: { type: "ephemeral" } },
  ];
  test.each(accepted.map((data) => [data]))("field is accepted: %p", (data) => {
    expect(validateBlock(AnthropicMessagesBlock, data) as unknown).toEqual(data);
  });

  const bad: unknown[] = [
    { output_config: { effort: "hgih" } },
    { thinking: { type: "enabled", budget_tokens: 100n } },
    { thinking: { type: "enabled" } },
    { thinking: { type: "adaptive", budget_tokens: 2048n } },
    { thinking: { type: "disabled", display: "omitted" } },
    { thinking: { type: "enabled", budget_tokens: 2048n, display: "full" } },
    { thinking: { type: "sometimes" } },
    { thinking: {} },
    { thinking: "enabled" },
    { max_tokens: 0n },
    { temperature: 1.5 },
    { top_p: -0.1 },
    { top_k: -1n },
    { service_tier: "priority" },
    { cache_control: { type: "ephemeral", ttl: "2h" } },
    { cache_control: { ttl: "5m" } },
    { metadata: { user_id: "u".repeat(513) } },
  ];
  test.each(bad.map((data) => [data]))("bad value is rejected: %p", (data) => {
    expect(() => validateBlock(AnthropicMessagesBlock, data)).toThrow(ValidationError);
  });

  test("thinking variant is picked by type for key checks", () => {
    let message = error(AnthropicMessagesBlock, { thinking: { type: "adaptive", budget_tokens: 2048n } });
    expect(message).toContain("`anthropic.thinking.budget_tokens` is not a supported field of the");
    expect(message).toContain("allowed: display, type");

    message = error(AnthropicMessagesBlock, { thinking: { type: "enabled", budget_token: 2048n } });
    expect(message).toContain("`anthropic.thinking.budget_token` is not a supported field of the");
    expect(message).toContain("allowed: budget_tokens, display, type");
  });

  test("thinking without a known type is left to validation", () => {
    const message = error(AnthropicMessagesBlock, { thinking: { type: SENTINEL, budget: 1n } });
    expect(message).not.toContain("`anthropic.thinking.budget`");
    expect(message).toContain("adaptive, between_tools, disabled, enabled");
    expect(message).not.toContain(SENTINEL);
    expect(error(AnthropicMessagesBlock, { thinking: { type: "constructor", budget: 1n } })).not.toContain(
      "`anthropic.thinking.budget`",
    );
  });

  test("a thinking variant's issues are under its tag", () => {
    expect(checkBlock(AnthropicMessagesBlock, { thinking: { type: "enabled" } })).toEqual({
      value: null,
      issues: [{ loc: ["thinking", "enabled", "budget_tokens"], type: "missing", msg: "Field required" }],
    });
  });

  const reserved: [unknown, string, string][] = [
    [{ output_config: { format: { type: "json_schema" } } }, "output_config.format", "structured output"],
    [{ mcp_servers: [] }, "mcp_servers", "tools"],
    [{ container: "c-1" }, "container", "tools"],
    [{ system: "be terse" }, "system", "system prompt"],
    [{ diagnostics: {} }, "diagnostics", "conversation state"],
    [{ messages: [] }, "messages", "messages"],
    [{ model: "claude" }, "model", "llm_config.model"],
    [{ tools: [] }, "tools", "tools"],
    [{ tool_choice: { type: "auto" } }, "tool_choice", "tools"],
    [{ stream: true }, "stream", "streaming"],
  ];
  test.each(reserved)("reserved path is built by yian: %p", (data, path, reason) => {
    expect(error(AnthropicMessagesBlock, data)).toContain(
      `\`anthropic.${path}\` is built by yian (${reason}); remove it`,
    );
  });

  test.each([["user_profile_id"], ["workspace_id"], ["temprature"]])(
    "fields outside the body list are unknown: %s",
    (key) => {
      expect(error(AnthropicMessagesBlock, { [key]: "x" })).toContain(
        `\`anthropic.${key}\` is not a supported field of the Anthropic Messages API request`,
      );
    },
  );

  test("the paths the Anthropic adapter builds are reserved", () => {
    for (const path of ["stream", "messages", "system", "tools", "tool_choice", "output_config.format"]) {
      expect(Object.hasOwn(AnthropicMessagesBlock.reserved, path)).toBe(true);
      expect(error(AnthropicMessagesBlock, nest(path, "x"))).toContain(`\`anthropic.${path}\` is built by yian (`);
    }
  });
});

describe("Gemini generateContent block", () => {
  test("valid sample dumps unchanged", () => {
    expect(validateBlock(GeminiGenerateContentBlock, GEMINI_SAMPLE)).toEqual(GEMINI_SAMPLE);
  });

  const accepted: unknown[] = [
    { generationConfig: { thinkingConfig: { thinkingLevel: "HIGH" } } },
    { generationConfig: { thinkingConfig: { thinkingBudget: 1024n } } },
    { generationConfig: { thinkingConfig: { thinkingBudget: -1n } } },
    { generationConfig: { thinkingConfig: { thinkingLevel: "MINIMAL" } } },
    { safetySettings: [{ category: "HARM_CATEGORY_JAILBREAK", threshold: "BLOCK_NONE" }] },
    { serviceTier: "priority" },
  ];
  test.each(accepted.map((data) => [data]))("field is accepted: %p", (data) => {
    expect(validateBlock(GeminiGenerateContentBlock, data) as unknown).toEqual(data);
  });

  const bad: unknown[] = [
    { generationConfig: { thinkingConfig: { thinkingLevel: "high" } } },
    { generationConfig: { thinkingConfig: { thinkingLevel: "THINKING_LEVEL_UNSPECIFIED" } } },
    { generationConfig: { thinkingConfig: { thinkingBudget: "1024" } } },
    { generationConfig: { temperature: 2.5 } },
    { generationConfig: { logprobs: 21n } },
    { generationConfig: { maxOutputTokens: 0n } },
    { generationConfig: { stopSequences: ["a", "b", "c", "d", "e", "f"] } },
    { generationConfig: { mediaResolution: "LOW" } },
    { safetySettings: [{ category: "HARM_CATEGORY_TOXICITY", threshold: "OFF" }] },
    { safetySettings: [{ category: "HARM_CATEGORY_HARASSMENT", threshold: "block_none" }] },
    { safetySettings: [{ category: "HARM_CATEGORY_HARASSMENT" }] },
    { safetySettings: { category: "HARM_CATEGORY_HARASSMENT", threshold: "OFF" } },
    { serviceTier: "FLEX" },
    { serviceTier: "unspecified" },
    { store: "no" },
  ];
  test.each(bad.map((data) => [data]))("bad value is rejected: %p", (data) => {
    expect(() => validateBlock(GeminiGenerateContentBlock, data)).toThrow(ValidationError);
  });

  test.each([["generation_config"], ["safety_settings"], ["service_tier"], ["labels"], ["temperature"]])(
    "snake-case and other keys are unknown: %s",
    (key) => {
      const message = error(GeminiGenerateContentBlock, { [key]: {} });
      expect(message).toContain(`\`google.${key}\` is not a supported field of the Gemini generateContent request`);
      expect(message).toContain("allowed: generationConfig, safetySettings, serviceTier, store");
    },
  );

  test.each([["responseModalities"], ["speechConfig"], ["imageConfig"], ["max_output_tokens"], ["thinking_config"]])(
    "generationConfig rejects unknown keys: %s",
    (key) => {
      const message = error(GeminiGenerateContentBlock, { generationConfig: { [key]: {} } });
      expect(message).toContain(`\`google.generationConfig.${key}\` is not a supported field of the`);
    },
  );

  test("a list item typo names its index", () => {
    const message = error(GeminiGenerateContentBlock, {
      safetySettings: [
        { category: "HARM_CATEGORY_HARASSMENT", threshold: "OFF" },
        { category: "HARM_CATEGORY_HATE_SPEECH", threshhold: "OFF" },
      ],
    });
    expect(message).toContain("`google.safetySettings[1].threshhold` is not a supported field of the");
    expect(message).toContain("allowed: category, threshold");
    expect(message).not.toContain("safetySettings[0]");
  });

  test("a nested typo names the full path", () => {
    const message = error(GeminiGenerateContentBlock, {
      generationConfig: { thinkingConfig: { thinking_level: "HIGH" } },
    });
    expect(message).toContain(
      "`google.generationConfig.thinkingConfig.thinking_level` is not a supported field of the",
    );
    expect(message).toContain("allowed: includeThoughts, thinkingBudget, thinkingLevel");
  });

  const reserved: [unknown, string, string][] = [
    [{ contents: [] }, "contents", "messages"],
    [{ systemInstruction: {} }, "systemInstruction", "system prompt"],
    [{ tools: [] }, "tools", "tools"],
    [{ toolConfig: {} }, "toolConfig", "tools"],
    [{ cachedContent: "cachedContents/1" }, "cachedContent", "system prompt and tools"],
    [{ continuationToken: "abc" }, "continuationToken", "conversation state"],
  ];
  test.each(reserved)("reserved path is built by yian: %p", (data, path, reason) => {
    expect(error(GeminiGenerateContentBlock, data)).toContain(
      `\`google.${path}\` is built by yian (${reason}); remove it`,
    );
  });

  const configReserved: [string, unknown, string][] = [
    ["responseSchema", {}, "structured output"],
    ["responseMimeType", "application/json", "structured output"],
    ["responseJsonSchema", {}, "structured output"],
    ["_responseJsonSchema", {}, "structured output"],
    ["responseFormat", {}, "structured output"],
    ["candidateCount", 2n, "yian reads one reply"],
  ];
  test.each(configReserved)("generationConfig reserved path is built by yian: %s", (key, value, reason) => {
    expect(error(GeminiGenerateContentBlock, { generationConfig: { [key]: value } })).toContain(
      `\`google.generationConfig.${key}\` is built by yian (${reason}); remove it`,
    );
  });
});

const ALL_BLOCKS: [string, BlockSchema][] = [
  ["AnthropicMessagesBlock", AnthropicMessagesBlock],
  ["GeminiGenerateContentBlock", GeminiGenerateContentBlock],
  ["OpenAIResponsesBlock", OpenAIResponsesBlock],
  ["OpenAISubscriptionBlock", OpenAISubscriptionBlock],
];

describe("every block", () => {
  test.each(ALL_BLOCKS)("%s: errors never echo values", (title, block) => {
    expect(block.title).toBe(title);
    const longValue = SENTINEL.repeat(30);
    const fields = Object.keys(block.fields);
    const each = (value: unknown) => Object.fromEntries(fields.map((name) => [name, value]));
    for (const data of [
      { typo: SENTINEL, model: SENTINEL, stream: [SENTINEL] },
      each(longValue),
      each([longValue]),
      each({ typo: longValue }),
      each({ type: longValue }),
    ]) {
      expect(error(block, data)).not.toContain(SENTINEL);
    }
  });

  test.each(ALL_BLOCKS)("%s: every reserved path is refused with its reason", (_title, block) => {
    for (const [path, reason] of Object.entries(block.reserved)) {
      const expected = `\`${block.name}.${path}\` is built by yian (${reason}); remove it`;
      expect(error(block, nest(path, "x"))).toContain(expected);
    }
  });

  test.each(ALL_BLOCKS)("%s: inherited object keys are unknown keys", (_title, block) => {
    const allowed = Object.keys(block.fields).sort(compareCodeUnits).join(", ");
    for (const key of ["constructor", "toString", "__proto__"]) {
      expect(error(block, Object.fromEntries([[key, "x"]]))).toContain(
        `\`${block.name}.${key}\` ${block.api}; allowed: ${allowed}`,
      );
    }
  });

  test("inherited object keys in a nested object are unknown keys", () => {
    for (const key of ["constructor", "toString", "__proto__"]) {
      expect(error(OpenAIResponsesBlock, { reasoning: Object.fromEntries([[key, "x"]]) })).toContain(
        `\`openai.reasoning.${key}\` is not a supported field of the OpenAI Responses API request; ` +
          "allowed: context, effort, mode, summary",
      );
    }
  });

  test("a __proto__ key in metadata is an own key of the result", () => {
    const result = validateBlock(OpenAIResponsesBlock, { metadata: Object.fromEntries([["__proto__", "v"]]) });
    const metadata = result["metadata"] as object;
    expect(Object.hasOwn(metadata, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(metadata)).toBe(Object.prototype);
    expect(Object.entries(metadata)).toEqual([["__proto__", "v"]]);
  });
});
