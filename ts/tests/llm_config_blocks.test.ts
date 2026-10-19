/**
 * Mirrors `tests/test_llm_config_blocks.py`: the `LLMConfig` block tests. The cascade merge tests
 * come with the next change to `config.ts` (`mergeLlmConfig`); the 3 `validate_layer` tests go
 * with `layers.ts`.
 */

import { describe, expect, test } from "bun:test";

import { LLM_CONFIG_KEYS, LLMConfig, SUPPORTED_LLM_PROVIDERS } from "../src/llm_clients/config.ts";
import { type Issue, PROVIDER_BLOCKS, ValidationError } from "../src/llm_clients/provider_blocks.ts";
import { fromHost } from "../src/typesys/values.ts";

const SENTINEL = "sk-sentinel-5f0c2a9e";

/** Return the `ValidationError` that building an `LLMConfig` from `data` throws. */
function failure(data: unknown): ValidationError {
  try {
    new LLMConfig(data);
  } catch (error) {
    expect(error).toBeInstanceOf(ValidationError);
    expect((error as ValidationError).title).toBe("LLMConfig");
    return error as ValidationError;
  }
  throw new Error("expected a ValidationError");
}

/** Return the message of the error that building an `LLMConfig` from `data` throws. */
function error(data: unknown): string {
  return failure(data).message;
}

/** Return the `{loc, type, msg}` of each issue found in `data`. */
function issues(data: unknown): Pick<Issue, "loc" | "type" | "msg">[] {
  return failure(data).issues.map(({ loc, type, msg }) => ({ loc, type, msg }));
}

test("every provider's block is checked by its own block schema", () => {
  for (const provider of SUPPORTED_LLM_PROVIDERS) {
    const block = PROVIDER_BLOCKS[provider]!;
    const caught = failure({ [provider]: { not_a_field: true } });
    expect(caught.issues.map((issue) => issue.loc)).toEqual([[provider]]);
    expect(caught.message).toContain(`\`${provider}.not_a_field\` ${block.api}`);
  }
});

test("llm config keys are spelled as authored", () => {
  expect(LLM_CONFIG_KEYS).toEqual(
    new Set([
      "provider",
      "model",
      "api_key",
      "base_url",
      "organization",
      "project",
      "headers",
      ...SUPPORTED_LLM_PROVIDERS,
    ]),
  );
  expect(LLM_CONFIG_KEYS.has("qwen_cn")).toBe(false);
});

test("block of the selected provider is valid", () => {
  const config = new LLMConfig({ provider: "anthropic", anthropic: { max_tokens: 8000n, temperature: 0.2 } });
  expect(config.blockFor("anthropic")["max_tokens"]).toBe(8000n);
  expect(config.blockFor("anthropic")).toEqual({ max_tokens: 8000n, temperature: 0.2 });
});

test("block of another provider is still checked", () => {
  const message = error({ provider: "anthropic", openai: { temprature: 0.2 } });
  expect(message).toContain("`openai.temprature` is not a supported field");
});

test("block contents are checked by type", () => {
  expect(error({ anthropic: { max_tokens: "8000" } })).toContain("anthropic.max_tokens");
});

test("dash block is accepted and dumped by its authored name", () => {
  const config = new LLMConfig({ provider: "qwen-cn", "qwen-cn": { enable_thinking: true } });
  expect(config.dump()["qwen-cn"]).toEqual({ enable_thinking: true });
  expect(Object.hasOwn(config.dump(), "qwen_cn")).toBe(false);
  expect(config.blockFor("qwen-cn")).toEqual({ enable_thinking: true });
});

test("dash block typo names the block", () => {
  expect(error({ "qwen-cn": { enable_thinkng: true } })).toContain("`qwen-cn.enable_thinkng` is not a supported field");
});

test.each([
  ["qwen_cn", "qwen-cn"],
  ["glm_cn", "glm-cn"],
  ["minimax_cn", "minimax-cn"],
])("python spelling %s of a dash block is rejected", (pythonName, authored) => {
  expect(error({ [pythonName]: {} })).toContain(`write \`${authored}\``);
});

test("keyword construction still works", () => {
  const config = new LLMConfig({ provider: "openai", model: "gpt-5.4", api_key: "sk-x" });
  expect(config.provider).toBe("openai");
  expect(config.apiKey).toBe("sk-x");
});

test("unset blocks are absent from every dump", () => {
  const dump = new LLMConfig({ provider: "openai", model: "gpt-5.4" }).dump();
  expect(Object.keys(dump).filter((key) => (SUPPORTED_LLM_PROVIDERS as readonly string[]).includes(key))).toEqual([]);
});

test("set block dumps without null fields at any depth", () => {
  const config = new LLMConfig({ openai: { reasoning: { effort: "high", summary: null }, temperature: null } });
  expect(config.dump()["openai"]).toEqual({ reasoning: { effort: "high" } });
});

test("blockFor an unset block is empty", () => {
  expect(new LLMConfig({ provider: "openai" }).blockFor("openai")).toEqual({});
  expect(new LLMConfig({ provider: "openai" }).blockFor("glm-cn")).toEqual({});
});

test("blockFor returns a fresh record", () => {
  const config = new LLMConfig({ "glm-cn": { thinking: { type: "enabled" } } });
  (config.blockFor("glm-cn")["thinking"] as Record<string, unknown>)["type"] = "disabled";
  expect(config.blockFor("glm-cn")).toEqual({ thinking: { type: "enabled" } });
});

test("dump round trips", () => {
  const config = new LLMConfig({
    provider: "google",
    model: "gemini-3-pro",
    google: { generationConfig: { temperature: 0.5, thinkingConfig: { thinkingLevel: "HIGH" } } },
    "minimax-cn": { thinking: { type: "adaptive" } },
    ollama: { think: true },
  });
  expect(new LLMConfig(config.dump()).dump()).toEqual(config.dump());
  expect(Object.keys(config.dump())).toEqual(["provider", "model", "minimax-cn", "ollama", "google"]);
});

test.each([
  [{ openai: { metadata: { team: 1n }, user: SENTINEL } }],
  [{ openai: { service_tier: SENTINEL } }],
  [{ anthropic: { max_tokens: SENTINEL } }],
  [{ "qwen-cn": { reasoning_effort: SENTINEL } }],
  [{ glm: SENTINEL }],
  [{ google: { safetySettings: [{ category: SENTINEL, threshold: "OFF" }] } }],
])("errors do not echo block values %#", (data) => {
  expect(error(data)).not.toContain(SENTINEL);
});

describe("TS additions", () => {
  test("a non-record is not a config", () => {
    for (const data of [[1], null, "provider: openai", new Map()]) {
      expect(issues(data)).toEqual([{ loc: [], type: "model_type", msg: "Input should be a valid dictionary" }]);
    }
  });

  test("a Python block name comes before a retired key, which comes before the Codex options", () => {
    expect(issues({ sandbox: 1n, temperature: 1n, qwen_cn: {} })).toEqual([
      { loc: [], type: "value_error", msg: "Value error, `qwen_cn` is not an llm_config key; write `qwen-cn`" },
    ]);
    expect(error({ sandbox: 1n, temperature: 1n })).toContain("`temperature` moved into the provider block");
    expect(error({ provider: "codex", temperature: 1n })).toContain("`temperature` moved into the provider block");
    expect(error({ provider: "CoDeX" })).toContain("Native Codex configuration has been removed");
    expect(error({ cwd: "/tmp" })).toContain("Native Codex configuration has been removed");
  });

  test("field issues come in field order, then extras in written order", () => {
    expect(
      issues({
        zzz: 1n,
        anthropic: 3n,
        headers: { "X-A": 1n },
        project: 3n,
        organization: "",
        base_url: "ftp://x",
        api_key: "",
        model: 1n,
        provider: "x",
        aaa: 2n,
        openai: { x: 1n },
      }).map(({ loc, type }) => [loc, type]),
    ).toEqual([
      [["provider"], "literal_error"],
      [["model"], "string_type"],
      [["api_key"], "value_error"],
      [["base_url"], "value_error"],
      [["organization"], "value_error"],
      [["project"], "string_type"],
      [["headers", "X-A"], "string_type"],
      [["openai"], "value_error"],
      [["anthropic"], "model_type"],
      [["zzz"], "extra_forbidden"],
      [["aaa"], "extra_forbidden"],
    ]);
  });

  test("the provider error lists the providers", () => {
    expect(issues({ provider: "nope" })).toEqual([
      {
        loc: ["provider"],
        type: "literal_error",
        msg:
          "Input should be 'openai', 'xai', 'deepseek', 'qwen', 'qwen-cn', 'glm', 'glm-cn', 'minimax', " +
          "'minimax-cn', 'openrouter', 'vllm', 'ollama', 'anthropic', 'google' or 'azure'",
      },
    ]);
  });

  test("field errors come before the connection rule", () => {
    expect(issues({ base_url: "https://a.example", model: 1n })).toEqual([
      { loc: ["model"], type: "string_type", msg: "Input should be a valid string" },
    ]);
  });

  test("value errors carry the validator's own message", () => {
    expect(failure({ provider: "openai", api_key: "" }).issues).toEqual([
      {
        loc: ["api_key"],
        type: "value_error",
        msg: "Value error, api_key must not be empty",
        error: "api_key must not be empty",
      },
    ]);
    const connection = failure({ base_url: "https://a.example" }).issues;
    expect(connection).toHaveLength(1);
    expect(connection[0]!.loc).toEqual([]);
    expect(connection[0]!.msg).toBe(`Value error, ${connection[0]!.error!}`);
    expect(connection[0]!.error).toContain("llm_config.base_url needs llm_config.api_key");
  });

  test("headers must be a record of strings", () => {
    expect(issues({ headers: ["X-A"] })).toEqual([
      { loc: ["headers"], type: "dict_type", msg: "Input should be a valid dictionary" },
    ]);
    expect(issues({ headers: { "A B": "1" } })).toEqual([
      {
        loc: ["headers"],
        type: "value_error",
        msg: "Value error, header names must be HTTP tokens: letters, digits and !#$%&'*+-.^_`|~",
      },
    ]);
    expect(new LLMConfig({ provider: "openai", headers: { "X-A": undefined } }).headers).toBeNull();
  });

  test("blockFor returns fresh copies and refuses a name with no block", () => {
    const config = new LLMConfig({ openai: { reasoning: { effort: "high" } } });
    expect(config.blockFor("openai")).not.toBe(config.blockFor("openai"));
    expect(config.dump()["openai"]).not.toBe(config.dump()["openai"]);
    for (const name of ["constructor", "toString", "qwen_cn", "codex"]) {
      expect(() => config.blockFor(name)).toThrow(RangeError);
    }
  });

  test("an instance is frozen", () => {
    const config = new LLMConfig({ provider: "openai", api_key: SENTINEL });
    expect(Object.isFrozen(config)).toBe(true);
    expect(() => {
      (config as { model: string | null }).model = "x";
    }).toThrow(TypeError);
    expect(config.apiKey).toBe(SENTINEL);
  });

  test("JSON.stringify shows exactly the five public fields", () => {
    const config = new LLMConfig({
      provider: "openai",
      model: "gpt-5.4",
      api_key: SENTINEL,
      openai: { max_output_tokens: 100n },
    });
    expect(JSON.parse(JSON.stringify(config))).toEqual({
      provider: "openai",
      model: "gpt-5.4",
      baseUrl: null,
      organization: null,
      project: null,
    });
    expect(JSON.stringify(config)).not.toContain(SENTINEL);
  });

  test("undefined properties read as absent", () => {
    const config = new LLMConfig({
      provider: "openai",
      api_key: undefined,
      temperature: undefined,
      qwen_cn: undefined,
    });
    expect(config.apiKey).toBeNull();
    expect(config.dump()).toEqual({ provider: "openai", model: null });
    expect(new LLMConfig({ openai: { stream: undefined } }).blockFor("openai")).toEqual({});
  });

  test("null counts as written in the presence checks", () => {
    expect(error({ temperature: null })).toContain("`temperature` moved into the provider block");
    expect(error({ sandbox: null })).toContain("Native Codex configuration has been removed");
    expect(error({ qwen_cn: null })).toContain("write `qwen-cn`");
    expect(error({ openai: { stream: null } })).toContain("`openai.stream` is built by yian");
    expect(issues({ zzz: null })).toEqual([
      { loc: ["zzz"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
    ]);
  });

  test("constructor and toString as top-level keys are extra inputs", () => {
    expect(issues({ constructor: "x", toString: "y" })).toEqual([
      { loc: ["constructor"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
      { loc: ["toString"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
    ]);
  });

  test("a fromHost config round-trips through dump", () => {
    const host = fromHost(
      { provider: "anthropic", model: "claude-sonnet-4-5", anthropic: { max_tokens: 8000, temperature: 0.5 } },
      "llm_config",
    );
    const config = new LLMConfig(host);
    expect(config.dump()).toEqual({
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      anthropic: { max_tokens: 8000n, temperature: 0.5 },
    });
    expect(new LLMConfig(config.dump()).dump()).toEqual(config.dump());
  });
});
