// Ported from tests/engine/test_cli_auth.py. Added here, since Python left them to the `cli`
// conformance cases: `yian providers` and `yian config check` (including `_config_errors`'s rows).
// The handlers are called directly with the `ParsedArgs` the parser would give, so parsing stays
// covered by cli_utils.test.ts and the `cli` cases.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { configCheck, configErrors, EXTRA_FIELD, providers } from "../../src/cli/auth.ts";
import { type CliIo, type ParsedArgs, UsageError, type Writer } from "../../src/cli/utils.ts";
import { LLM_CONFIG_KEYS } from "../../src/llm_clients/config.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-auth-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** A writer collecting what it is given. */
function sink() {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  return { out, text: () => chunks.join("") };
}

/**
 * A command's I/O: stdin ended with `stdin` (text or bytes) and marked `isTTY` as given, collected
 * stdout and stderr, and `env` as the environment.
 */
function fakeIo(options: { stdin?: string | Uint8Array; isTTY?: boolean; env?: Record<string, string> } = {}) {
  const stdin = Object.assign(new PassThrough(), { isTTY: options.isTTY });
  stdin.end(options.stdin === undefined ? undefined : Buffer.from(options.stdin));
  const stdout = sink();
  const stderr = sink();
  const env = options.env ?? {};
  const io: CliIo = {
    stdin,
    stdout: stdout.out,
    stderr: stderr.out,
    env: (name) => (Object.hasOwn(env, name) ? env[name] : undefined),
  };
  return { io, out: stdout.text, err: stderr.text };
}

/** The `ParsedArgs` of a command with no positionals, `--json` as given. */
function jsonArgs(json: boolean): ParsedArgs {
  return { help: false, args: {}, options: { json } };
}

type Handler<D> = (command: ParsedArgs, io: CliIo, deps?: D) => Promise<number>;

/**
 * Run `handler` and return its exit code and output; a `UsageError` gives code 2 and its message
 * on stderr, as `cli/index.ts` prints it after the usage.
 */
async function run<D>(handler: Handler<D>, args: ParsedArgs, io: ReturnType<typeof fakeIo>, deps?: D) {
  let code: number;
  try {
    code = await handler(args, io.io, deps);
  } catch (error) {
    if (!(error instanceof UsageError)) throw error;
    await io.io.stderr.write(`Error: ${error.message}\n`);
    code = 2;
  }
  return { code, out: io.out(), err: io.err() };
}

const PROVIDERS_TEXT = `openai: api-key, subscription (OPENAI_API_KEY)
xai: api-key (XAI_API_KEY)
deepseek: api-key (DEEPSEEK_API_KEY)
qwen: api-key (DASHSCOPE_API_KEY)
qwen-cn: api-key (DASHSCOPE_CN_API_KEY)
glm: api-key (ZHIPU_API_KEY)
glm-cn: api-key (ZHIPU_CN_API_KEY)
minimax: api-key (MINIMAX_API_KEY)
minimax-cn: api-key (MINIMAX_CN_API_KEY)
openrouter: api-key (OPENROUTER_API_KEY)
vllm: api-key, none (VLLM_API_KEY)
ollama: none
anthropic: api-key (ANTHROPIC_API_KEY)
google: api-key (GOOGLE_API_KEY)
azure: api-key (AZURE_OPENAI_API_KEY)
`;

describe("providers", () => {
  test("text lists every provider, its login kinds and its key variable", async () => {
    const result = await run(providers, jsonArgs(false), fakeIo());
    expect(result).toEqual({ code: 0, out: PROVIDERS_TEXT, err: "" });
  });

  test("JSON lists the same providers in the same order", async () => {
    const result = await run(providers, jsonArgs(true), fakeIo());
    expect(result.code).toBe(0);
    const listed = PROVIDERS_TEXT.trimEnd()
      .split("\n")
      .map((line) => {
        const [, name, methods, variable] = /^(\S+): ([^(]+?)(?: \((\w+)\))?$/.exec(line)!;
        return { name, methods: methods!.split(", "), api_key_env: variable ?? null };
      });
    expect(listed).toHaveLength(15);
    expect(JSON.parse(result.out)).toEqual({ format: "yian.cli.v1", providers: listed });
  });
});

describe("configErrors", () => {
  test("a valid layer has none", () => {
    expect(configErrors({ provider: "openai", model: "gpt-5.5", api_key: "k" })).toEqual([]);
  });

  test("a provider block reports its field", () => {
    const errors = configErrors({ provider: "openai", openai: { temperature: "hot" } });
    expect(errors).toEqual([{ field: "openai.temperature", message: "Input should be a valid number" }]);
  });

  test("a key without a provider has no field", () => {
    const errors = configErrors({ api_key: "k" });
    expect(errors).toEqual([
      { field: null, message: "llm_config.api_key needs llm_config.provider, so the key goes only to that provider" },
    ]);
  });

  test("an extra top-level key is reported without its name", () => {
    const errors = configErrors({ provider: "openai", "sk-pasted-key": 1 });
    expect(errors).toEqual([{ field: null, message: EXTRA_FIELD }]);
    expect(EXTRA_FIELD).not.toContain("sk-pasted-key");
    expect(EXTRA_FIELD).toBe(`llm_config has an extra field; the fields are ${[...LLM_CONFIG_KEYS].sort().join(", ")}`);
  });

  test("a bad base_url with a key names base_url", () => {
    const errors = configErrors({ provider: "openai", api_key: "k", base_url: "http://example.com/v1" });
    expect(errors.map((error) => error.field)).toEqual(["base_url"]);
  });

  test("identical errors are listed once", () => {
    expect(configErrors({ provider: "openai", a: 1, b: 2 })).toEqual([{ field: null, message: EXTRA_FIELD }]);
  });

  test("messages drop pydantic's value error prefix", () => {
    const errors = configErrors({ provider: "openai", openai: { tools: [] } });
    expect(errors).toHaveLength(1);
    expect(errors[0]!.field).toBe("openai");
    expect(errors[0]!.message).toStartWith("`openai.tools` is built by yian");
    expect(errors.some((error) => error.message.startsWith("Value error, "))).toBe(false);
  });

  test("the layer given is not changed", () => {
    const layer = { provider: "openai", api_key: "k" };
    configErrors(layer);
    expect(layer).toEqual({ provider: "openai", api_key: "k" });
  });
});

describe("configCheck", () => {
  const notObject = "Invalid value: stdin must be a JSON object (an llm_config layer)";

  test.each([
    ["not JSON", "provider: openai"],
    ["a list", "[1]"],
    ["NaN", "NaN"],
    ["a BOM before the object", "\uFEFF{}"],
    ["bytes that aren't UTF-8", new Uint8Array([0x7b, 0xff, 0x7d])],
  ])("refuses stdin that is %s", async (_, stdin) => {
    const io = fakeIo({ stdin });
    await expect(configCheck(jsonArgs(true), io.io)).rejects.toThrow(new UsageError(notObject));
    expect(io.out()).toBe("");
  });

  test("text prints ok for a valid layer", async () => {
    const io = fakeIo({ stdin: '{"provider": "openai", "model": "gpt-5.5", "api_key": "k"}' });
    expect(await run(configCheck, jsonArgs(false), io)).toEqual({ code: 0, out: "ok\n", err: "" });
  });

  test("text prints one line per error on stderr and exits 1", async () => {
    const io = fakeIo({ stdin: '{"provider": "openai", "openai": {"temperature": "hot"}, "x": 1}' });
    expect(await run(configCheck, jsonArgs(false), io)).toEqual({
      code: 1,
      out: "",
      err: `openai.temperature: Input should be a valid number\n${EXTRA_FIELD}\n`,
    });
  });

  test("JSON prints ok and the errors", async () => {
    const valid = await run(configCheck, jsonArgs(true), fakeIo({ stdin: '{"provider": "openai"}' }));
    expect(valid.code).toBe(0);
    expect(JSON.parse(valid.out)).toEqual({ format: "yian.cli.v1", ok: true, errors: [] });
    const invalid = await run(configCheck, jsonArgs(true), fakeIo({ stdin: '{"api_key": "k"}' }));
    expect(invalid.code).toBe(1);
    expect(JSON.parse(invalid.out)).toEqual({
      format: "yian.cli.v1",
      ok: false,
      errors: [
        { field: null, message: "llm_config.api_key needs llm_config.provider, so the key goes only to that provider" },
      ],
    });
  });
});
