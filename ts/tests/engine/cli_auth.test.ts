// Ported from tests/engine/test_cli_auth.py: `test_models_refuses_stdin_that_is_not_text`. Added
// here, since Python left them to the `cli` conformance cases: `yian providers`, `yian config check`
// (including `_config_errors`'s rows) and the rest of `yian models` (usage errors, the provider and
// subscription forms, the deadline). Python's monkeypatched `list_models` and `OpenAIAuth` are
// `deps` fakes here, so no test reaches a real host.
// The handlers are called directly with the `ParsedArgs` the parser would give, so parsing stays
// covered by cli_utils.test.ts and the `cli` cases.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { type AuthDeps, configCheck, configErrors, EXTRA_FIELD, models, providers } from "../../src/cli/auth.ts";
import { type CliIo, type OptionValue, type ParsedArgs, UsageError, type Writer } from "../../src/cli/utils.ts";
import { ApiKeyCredentials, AuthError, AuthStore, OAuthCredentials } from "../../src/llm_clients/auth/index.ts";
import { LLM_CONFIG_KEYS, SUPPORTED_LLM_PROVIDERS } from "../../src/llm_clients/config.ts";
import { ModelListError } from "../../src/llm_clients/model_list.ts";
import { virtualScheduler } from "../../src/platform.ts";

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

/** The `ParsedArgs` of `yian models` with `options`: `--json` absent is `false`. */
function modelsArgs(options: Record<string, OptionValue> = {}): ParsedArgs {
  return { help: false, args: {}, options: { json: false, ...options } };
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

/** A fake `listModels` answering with `answer` (a list, `null`, or an error to throw) and recording each call. */
function fakeListModels(answer: readonly string[] | null | Error = ["m-b", "m-a"]) {
  const calls: { provider: unknown; options: Record<string, unknown> }[] = [];
  const listModels = (async (provider: unknown, options: Record<string, unknown> = {}) => {
    calls.push({ provider, options });
    if (answer instanceof Error) throw answer;
    return answer;
  }) as NonNullable<AuthDeps["listModels"]>;
  return { listModels, calls };
}

describe("models usage", () => {
  test("refuses stdin that is not text", async () => {
    const fake = fakeListModels();
    const io = fakeIo({ stdin: new Uint8Array([0xff, 0xfe]) });
    const result = await run(models, modelsArgs({ provider: "openrouter", json: true }), io, fake);
    expect(result.code).toBe(2);
    expect(result.out).toBe("");
    expect(result.err).toContain("Invalid value: stdin must be the API key as text");
    expect(fake.calls).toEqual([]);
  });

  const exactlyOne = "Invalid value: Give exactly one of --auth-file and --provider";
  const timeout = "Invalid value: --timeout must be a finite number of seconds above 0";
  test.each<[string, Record<string, OptionValue>, string]>([
    ["neither source", {}, exactlyOne],
    ["both sources", { "auth-file": "auth.json", provider: "openai" }, exactlyOne],
    ["--base-url with --auth-file", { "auth-file": "auth.json", "base-url": "https://gateway.example/v1" }, "Invalid value: --base-url applies only to --provider"],
    ["an unsupported provider", { provider: "nope" }, `Invalid value: Unsupported provider; choose one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`],
    ["--timeout 0", { provider: "openrouter", timeout: 0 }, timeout],
    ["--timeout=-1", { provider: "openrouter", timeout: -1 }, timeout],
    ["--timeout inf", { provider: "openrouter", timeout: Infinity }, timeout],
    ["--timeout nan", { provider: "openrouter", timeout: NaN }, timeout],
  ])("refuses %s", async (_, options, message) => {
    const fake = fakeListModels();
    let built = false;
    const deps: AuthDeps = {
      listModels: fake.listModels,
      openAIAuth: () => {
        built = true;
        throw new Error("built a service");
      },
    };
    const io = fakeIo({ stdin: "k" });
    await expect(models(modelsArgs({ json: true, ...options }), io.io, deps)).rejects.toThrow(new UsageError(message));
    expect(io.out()).toBe("");
    expect(fake.calls).toEqual([]);
    expect(built).toBe(false);
  });
});

describe("models --provider", () => {
  test("sends the stripped stdin key, the lowercased provider and the default timeout", async () => {
    const fake = fakeListModels();
    const result = await run(models, modelsArgs({ provider: "OpenRouter", json: true }), fakeIo({ stdin: " k\n" }), fake);
    expect(result.code).toBe(0);
    expect(JSON.parse(result.out)).toEqual({ format: "yian.cli.v1", models: ["m-b", "m-a"] });
    expect(fake.calls).toEqual([{ provider: "openrouter", options: { apiKey: "k", baseUrl: null, timeout: 8 } }]);
  });

  test.each<[string, { stdin?: string; isTTY?: boolean }]>([
    ["empty stdin", { stdin: "" }],
    ["blank stdin", { stdin: " \n" }],
    ["a terminal, which isn't read", { stdin: "k", isTTY: true }],
  ])("%s sends no key", async (_, stdin) => {
    const fake = fakeListModels();
    await run(models, modelsArgs({ provider: "ollama" }), fakeIo(stdin), fake);
    expect(fake.calls[0]!.options["apiKey"]).toBeNull();
  });

  test("passes --timeout and --base-url through", async () => {
    const fake = fakeListModels();
    const args = modelsArgs({ provider: "openrouter", "base-url": "https://gateway.example/api/v1", timeout: 0.5 });
    await run(models, args, fakeIo({ stdin: "k" }), fake);
    expect(fake.calls[0]!.options).toEqual({ apiKey: "k", baseUrl: "https://gateway.example/api/v1", timeout: 0.5 });
  });

  test.each<[string, string, bigint | null]>([
    ["HTTP 403", "models.failed", 403n],
    ["timed out", "models.timeout", null],
    ["can't reach host", "models.failed", null],
  ])("a listing failing with %s is %s", async (reason, code, status) => {
    const fake = fakeListModels(new ModelListError(reason));
    const result = await run(models, modelsArgs({ provider: "openrouter", json: true }), fakeIo({ stdin: "k" }), fake);
    expect(result.code).toBe(1);
    const message = `Couldn't list the openrouter models: ${reason}`;
    expect(JSON.parse(result.out)).toEqual({
      format: "yian.cli.v1",
      error: { code, message, status: status === null ? null : Number(status) },
    });
    const text = await run(models, modelsArgs({ provider: "openrouter" }), fakeIo({ stdin: "k" }), fake);
    expect(text).toEqual({ code: 1, out: "", err: `${message}\n` });
  });

  test("the JSON status is an int", async () => {
    const fake = fakeListModels(new ModelListError("HTTP 403"));
    const result = await run(models, modelsArgs({ provider: "openrouter", json: true }), fakeIo({ stdin: "k" }), fake);
    expect(result.out).toContain('"status":403}');
  });

  test("azure has no listing", async () => {
    const fake = fakeListModels(null);
    const json = await run(models, modelsArgs({ provider: "azure", json: true }), fakeIo({ stdin: "k" }), fake);
    expect(json.code).toBe(0);
    expect(JSON.parse(json.out)).toEqual({ format: "yian.cli.v1", models: null });
    const text = await run(models, modelsArgs({ provider: "azure" }), fakeIo({ stdin: "k" }), fake);
    expect(text).toEqual({ code: 0, out: "", err: "azure has no model listing\n" });
  });

  test("text prints one id per line", async () => {
    const fake = fakeListModels(["m-a", "m-b"]);
    const result = await run(models, modelsArgs({ provider: "openrouter" }), fakeIo({ stdin: "k" }), fake);
    expect(result).toEqual({ code: 0, out: "m-a\nm-b\n", err: "" });
  });
});

describe("models --auth-file", () => {
  const FAR = 4102444800;

  /** OAuth credentials for `account-1` with `accessToken`, expiring at `expiresAt` (epoch seconds). */
  function oauth(accessToken: string, expiresAt = FAR) {
    return new OAuthCredentials({ accessToken, refreshToken: `${accessToken}-refresh`, expiresAt, accountId: "account-1" });
  }

  /**
   * A fake `openAIAuth` factory recording the options each service is built with; its services
   * list with `list` and refresh to a fresh `new-access` token, recording both calls.
   */
  function fakeServices(list: (credentials: OAuthCredentials, signal?: AbortSignal) => Promise<readonly string[]>) {
    const built: { signal?: AbortSignal; requestTimeoutMs?: number }[] = [];
    const calls: string[] = [];
    const openAIAuth: AuthDeps["openAIAuth"] = (options) => {
      built.push(options);
      return {
        refresh: async (old: OAuthCredentials) => {
          calls.push(`refresh ${old.accessToken}`);
          return oauth("new-access");
        },
        subscriptionModels: async (credentials: OAuthCredentials) => {
          calls.push(`list ${credentials.accessToken}`);
          return list(credentials, options.signal);
        },
      };
    };
    return { openAIAuth, built, calls };
  }

  /** The path of an auth file holding a subscription login with `credentials`. */
  async function savedLogin(credentials: OAuthCredentials | ApiKeyCredentials = oauth("old-access")) {
    const path = join(tmp, "auth.json");
    await new AuthStore(path).save("openai", credentials, "gpt-a");
    return path;
  }

  /** Run `yian models --auth-file path --json` with `options` and return the code and document. */
  async function listJson(path: string, deps: AuthDeps, options: Record<string, OptionValue> = {}) {
    const result = await run(models, modelsArgs({ "auth-file": path, json: true, ...options }), fakeIo(), deps);
    return { code: result.code, document: JSON.parse(result.out), err: result.err };
  }

  test("lists the subscription models", async () => {
    const path = await savedLogin();
    const fake = fakeServices(async () => ["gpt-a", "gpt-b"]);
    const result = await listJson(path, fake);
    expect(result).toEqual({ code: 0, document: { format: "yian.cli.v1", models: ["gpt-a", "gpt-b"] }, err: "" });
    expect(fake.calls).toEqual(["list old-access"]);
    const text = await run(models, modelsArgs({ "auth-file": path }), fakeIo(), fake);
    expect(text).toEqual({ code: 0, out: "gpt-a\ngpt-b\n", err: "" });
  });

  test.each([
    ["no auth file", false],
    ["an API-key openai login", true],
  ])("%s has no subscription login", async (_, saved) => {
    const path = saved ? await savedLogin(new ApiKeyCredentials("k")) : join(tmp, "missing", "auth.json");
    const fake = fakeServices(async () => ["gpt-a"]);
    const { code, document } = await listJson(path, fake);
    expect(code).toBe(1);
    expect(document.error).toEqual({
      code: "models.failed",
      message: `Couldn't list the openai subscription models: There is no saved openai subscription login in ${path}`,
      status: null,
    });
    expect(fake.calls).toEqual([]);
  });

  test.each<[string, number | null]>([
    ["OpenAI authentication request failed (HTTP 401)", 401],
    ["OpenAI returned an invalid model catalog", null],
  ])("a refused listing (%s) is models.failed", async (message, status) => {
    const path = await savedLogin();
    const fake = fakeServices(async () => {
      throw new AuthError(message);
    });
    const { code, document } = await listJson(path, fake);
    expect(code).toBe(1);
    expect(document.error).toEqual({
      code: "models.failed",
      message: `Couldn't list the openai subscription models: ${message}`,
      status,
    });
  });

  test("a listing past --timeout is models.timeout", async () => {
    const path = await savedLogin();
    const fake = fakeServices(
      (_, signal) => new Promise((_, reject) => signal!.addEventListener("abort", () => reject(signal!.reason))),
    );
    const { code, document } = await listJson(path, fake, { timeout: 0.05 });
    expect(code).toBe(1);
    expect(document.error).toEqual({
      code: "models.timeout",
      message: "Couldn't list the openai subscription models: timed out",
      status: null,
    });
  });

  test("the service gets the deadline's signal and --timeout as its request timeout", async () => {
    const path = await savedLogin();
    const fake = fakeServices(async () => ["gpt-a"]);
    await listJson(path, fake, { timeout: 0.5 });
    expect(fake.built).toHaveLength(1);
    expect(fake.built[0]!.requestTimeoutMs).toBe(500);
    expect(fake.built[0]!.signal).toBeInstanceOf(AbortSignal);
  });

  test("an expiring token is refreshed and saved first", async () => {
    const path = await savedLogin(oauth("old-access", Date.now() / 1000 + 10));
    const fake = fakeServices(async () => ["gpt-a"]);
    const { code } = await listJson(path, fake);
    expect(code).toBe(0);
    expect(fake.calls).toEqual(["refresh old-access", "list new-access"]);
    const store = new AuthStore(path);
    const saved = await store.credentials("openai", store.selection("openai")!.credentialId);
    expect((saved as OAuthCredentials).accessToken).toBe("new-access");
  });

  test.each([1e9, 1e306])("--timeout %p neither aborts nor throws", async (timeout) => {
    const path = await savedLogin();
    const fake = fakeServices(
      (_, signal) =>
        new Promise((resolve, reject) => {
          signal?.addEventListener("abort", () => reject(signal.reason));
          setTimeout(() => resolve(["gpt-a"]), 20);
        }),
    );
    const { code, document } = await listJson(path, fake, { timeout });
    expect(code).toBe(0);
    expect(document.models).toEqual(["gpt-a"]);
  });

  test("no deadline sleeper outlives a listing", async () => {
    const path = await savedLogin();
    const scheduler = virtualScheduler();
    const fake = fakeServices(async () => ["gpt-a"]);
    const { code } = await listJson(path, { ...fake, scheduler });
    expect(code).toBe(0);
    // A sleeper left behind would move the virtual clock to the 8 s deadline meanwhile.
    await new Promise((resolve) => setTimeout(resolve, 10));
    await scheduler.sleep(1);
    expect(scheduler.now()).toBe(1);
  });
});
