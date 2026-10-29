// `yian auth`, `yian models`, `yian providers` and `yian config check`: sign-in and setup, saved
// logins and their removal, the model listing and the one-reply check, secrets scrubbed from
// messages, and SIGINT at each step.
// The model listing, OpenAI sign-in service, chat client, pickers, terminal and browser are `deps`
// fakes (a fake terminal, a fake chat model, a recording opener and `launch`), so no test reaches
// a real host or opens a browser.
// The handlers are called directly with the `ParsedArgs` the parser would give, so parsing stays
// covered by cli_utils.test.ts and the `cli` cases.

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { EventEmitter } from "node:events";
import fs, { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import {
  auth,
  type AuthDeps,
  configCheck,
  configErrors,
  EXTRA_FIELD,
  modelAnswers,
  models,
  openBrowser,
  providers,
} from "../../src/cli/auth.ts";
import { type CliIo, type OptionValue, type ParsedArgs, UsageError, type Writer } from "../../src/cli/utils.ts";
import { PROVIDER_API_KEY_ENV } from "../../src/llm_clients/api_key_env.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type AuthRequest,
  AuthStore,
  HttpResponse,
  OAuthCredentials,
  OpenAIAuth,
} from "../../src/llm_clients/auth/index.ts";
import { LLM_CONFIG_KEYS, SUPPORTED_LLM_PROVIDERS, validateBaseUrl } from "../../src/llm_clients/config.ts";
import { ConnectionSnapshot } from "../../src/llm_clients/connection.ts";
import {
  OPENAI_REFUSED_HEADER_NAMES,
  REFUSED_HEADER_NAMES,
  validateHeaderName,
  validateHeaderValue,
  validateOrganization,
} from "../../src/llm_clients/connection_settings.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import { type ListModelsOptions, listModels, ModelListError } from "../../src/llm_clients/model_list.ts";
import { type Event, Finish, PartDone, TextDelta } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { lockFile, virtualScheduler } from "../../src/platform.ts";

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
 * stdout and stderr (TTYs when `outTty`), and `env` as the environment.
 */
function fakeIo(
  options: { stdin?: string | Uint8Array; isTTY?: boolean; outTty?: boolean; env?: Record<string, string> } = {},
) {
  const stdin = Object.assign(new PassThrough(), { isTTY: options.isTTY });
  stdin.end(options.stdin === undefined ? undefined : Buffer.from(options.stdin));
  const stdout = sink();
  const stderr = sink();
  const env = options.env ?? {};
  const io: CliIo = {
    stdin,
    stdout: { ...stdout.out, isTTY: options.outTty ?? false },
    stderr: { ...stderr.out, isTTY: options.outTty ?? false },
    stderrStream: new PassThrough(),
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

const CANCELLED = "Authentication cancelled; the previous login is unchanged";

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
google: api-key (GOOGLE_API_KEY or GEMINI_API_KEY)
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
        // The JSON field holds only the first key variable.
        const [, name, methods, variable] = /^(\S+): ([^(]+?)(?: \((\w+)(?: or \w+)*\))?$/.exec(line)!;
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

  test("config messages carry no \"Value error, \" prefix", () => {
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
    ["an object with a repeated key", '{"provider": "openai", "provider": "xai"}'],
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

  /** A `listModels` over a fake transport that records each requested URL and lists `gateway-model`. */
  function recordedListing() {
    const urls: string[] = [];
    const listing: AuthDeps["listModels"] = (provider, options) =>
      listModels(provider, {
        ...options,
        request: async (_method, url) => {
          urls.push(url);
          return new HttpResponse(200, { data: [{ id: "gateway-model" }] });
        },
      });
    return { urls, listing };
  }

  test("openai lists OPENAI_BASE_URL when it is set and sends nothing to api.openai.com", async () => {
    const recorded = recordedListing();
    const io = fakeIo({ stdin: "private-key", env: { OPENAI_BASE_URL: "https://gateway.example.com/v1" } });
    const result = await run(models, modelsArgs({ provider: "openai" }), io, { listModels: recorded.listing });
    expect(result).toMatchObject({ code: 0, out: "gateway-model\n" });
    expect(recorded.urls).toEqual(["https://gateway.example.com/v1/models"]);
    expect(recorded.urls.some((url) => url.includes("api.openai.com"))).toBe(false);
  });

  test("an invalid OPENAI_BASE_URL fails the openai listing without a request", async () => {
    const recorded = recordedListing();
    const io = fakeIo({ stdin: "private-key", env: { OPENAI_BASE_URL: "http://gateway.example.com/v1" } });
    const result = await run(models, modelsArgs({ provider: "openai" }), io, { listModels: recorded.listing });
    expect(result.code).toBe(1);
    expect(result.err).toContain("Couldn't list the openai models: invalid request");
    expect(result.out + result.err).not.toContain("private-");
    expect(recorded.urls).toEqual([]);
  });

  test("--base-url wins over OPENAI_BASE_URL, and other providers ignore OPENAI_BASE_URL", async () => {
    const fake = fakeListModels();
    const env = { OPENAI_BASE_URL: "https://gateway.example.com/v1" };
    const explicit = modelsArgs({ provider: "openai", "base-url": "https://other.example/v1" });
    await run(models, explicit, fakeIo({ stdin: "k", env }), fake);
    await run(models, modelsArgs({ provider: "openrouter" }), fakeIo({ stdin: "k", env }), fake);
    expect(fake.calls.map((call) => call.options["baseUrl"])).toEqual(["https://other.example/v1", null]);
  });

  test.each<[string, string, bigint | null]>([
    ["HTTP 403", "models.failed", 403n],
    ["timed out", "models.timeout", null],
    ["can't reach host", "models.failed", null],
  ])("a listing failing with %s is %s", async (reason, code, status) => {
    // As `listModels` throws them: an HTTP refusal carries its status, a timeout its flag.
    const failure = new ModelListError(reason, {
      ...(status === null ? {} : { status: Number(status) }),
      timedOut: reason === "timed out",
    });
    const fake = fakeListModels(failure);
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

  test("a listing's timeout is known by its flag, not its message", async () => {
    const fake = fakeListModels(new ModelListError("deadline passed", { timedOut: true }));
    const result = await run(models, modelsArgs({ provider: "openrouter", json: true }), fakeIo({ stdin: "k" }), fake);
    expect(JSON.parse(result.out).error.code).toBe("models.timeout");
  });

  test("the JSON status is an int", async () => {
    const fake = fakeListModels(new ModelListError("HTTP 403", { status: 403 }));
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

describe("models --provider with escape sequences in the provider's text", () => {
  const raw = "m\x1b]52;c;aGk=\x07-a\x1b[2J";

  test("a model id is stripped on a TTY stdout and kept exactly when piped", async () => {
    const fake = fakeListModels([raw]);
    const tty = await run(models, modelsArgs({ provider: "openrouter" }), fakeIo({ stdin: "k", outTty: true }), fake);
    expect(tty).toEqual({ code: 0, out: "m-a\n", err: "" });
    const piped = await run(models, modelsArgs({ provider: "openrouter" }), fakeIo({ stdin: "k" }), fake);
    expect(piped).toEqual({ code: 0, out: `${raw}\n`, err: "" });
  });

  test("a listing failure is stripped on a TTY stderr and kept exactly when piped", async () => {
    const fake = fakeListModels(new ModelListError(`HTTP 400: ${raw}`, { status: 400 }));
    const args = modelsArgs({ provider: "openrouter" });
    const tty = await run(models, args, fakeIo({ stdin: "k", outTty: true }), fake);
    expect(tty.err).toBe("Couldn't list the openrouter models: HTTP 400: m-a\n");
    const piped = await run(models, args, fakeIo({ stdin: "k" }), fake);
    expect(piped.err).toBe(`Couldn't list the openrouter models: HTTP 400: ${raw}\n`);
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
        deviceLogin: async () => {
          throw new Error("yian models never signs in");
        },
        refresh: async (old: OAuthCredentials) => {
          calls.push(`refresh ${old.accessToken}`);
          return oauth("new-access");
        },
        subscriptionModels: async (credentials: OAuthCredentials) => {
          calls.push(`list ${credentials.accessToken}`);
          return list(credentials, options.signal);
        },
        browserLogin: async () => {
          throw new Error("yian models never signs in");
        },
        apiModels: async () => {
          throw new Error("yian models never lists API-key models");
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
    // As `OpenAIAuth` throws them: an HTTP refusal carries its status.
    const fake = fakeServices(async () => {
      throw new AuthError(message, status === null ? {} : { status });
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
      // A request cut short rejects with a plain AuthError, not the deadline's reason.
      (_, signal) =>
        new Promise((_, reject) =>
          signal!.addEventListener("abort", () => reject(new AuthError("OpenAI authentication request failed"))),
        ),
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

/** The `ParsedArgs` of `yian auth [provider]` with `options`: flags absent are `false`, `--header` absent `[]`. */
function authArgs(provider?: string, options: Record<string, OptionValue> = {}): ParsedArgs {
  return {
    help: false,
    args: provider === undefined ? {} : { provider },
    options: { header: [], default: false, remove: false, json: false, ...options },
  };
}

/**
 * A fake OpenAI authentication service recording each call, and the key,
 * headers and settings each API-key listing is sent. Its browser login opens a fixed URL. Its
 * factory records the options it is given in `built`.
 */
function fakeService() {
  const calls: string[] = [];
  const built: Parameters<NonNullable<AuthDeps["openAIAuth"]>>[0][] = [];
  const settings: Record<string, unknown>[] = [];
  const credentials = () =>
    new OAuthCredentials({
      accessToken: "private-access",
      refreshToken: "private-refresh",
      expiresAt: 9999999999,
      accountId: "test-account",
    });
  const service = {
    deviceLogin: async (display: (url: string, code: string) => void | Promise<void>) => {
      calls.push("device");
      await display("https://example.invalid/device", "TEST-CODE");
      return credentials();
    },
    browserLogin: async (openUrl: (url: string) => void | Promise<void>) => {
      calls.push("browser");
      await openUrl("https://example.invalid/authorize");
      return credentials();
    },
    refresh: async (old: OAuthCredentials) => {
      calls.push("refresh");
      return old;
    },
    subscriptionModels: async () => {
      calls.push("subscription_models");
      return ["gpt-test", "gpt-other"];
    },
    apiModels: async (
      sent: ApiKeyCredentials,
      options: { organization?: unknown; project?: unknown } = {},
    ): Promise<readonly string[]> => {
      calls.push("api_models");
      settings.push({ apiKey: sent.apiKey, headers: sent.headers, ...options });
      return ["embedding-test", "gpt-test", "gpt-other"];
    },
  };
  const openAIAuth: NonNullable<AuthDeps["openAIAuth"]> = (options) => {
    built.push(options);
    return service;
  };
  return { calls, settings, built, service, openAIAuth };
}

/**
 * A store path in `YIAN_AUTH_FILE`, `OPENAI_API_KEY=private-key`, the suite's environment
 * defaults, and a fake service. The store's
 * directory is private (`storeOpen`, the default), or 0755, so any store read fails with
 * "Credential storage must be private to its owner". The browser opener records each URL in
 * `opened`.
 */
function setupAuth({ storeOpen = true }: { storeOpen?: boolean } = {}) {
  const directory = join(tmp, "auth");
  const mode = storeOpen ? 0o700 : 0o755;
  mkdirSync(directory, { mode });
  chmodSync(directory, mode);
  const path = join(directory, "auth.json");
  const env: Record<string, string> = {
    YIAN_AUTH_FILE: path,
    OPENAI_API_KEY: "private-key",
    YIAN_DEFAULT_PROVIDER: "openai",
    YIAN_DEFAULT_MODEL: "gpt-5.5",
  };
  const service = fakeService();
  // Recorded, so no test opens a real browser.
  const opened: string[] = [];
  const deps: AuthDeps = { openAIAuth: service.openAIAuth, openUrl: (url) => void opened.push(url) };
  return { path, store: new AuthStore(path), env, service, deps, opened };
}

/** The message of the `RangeError` `check` throws. */
function rangeMessage(check: () => unknown): string {
  try {
    check();
  } catch (error) {
    if (error instanceof RangeError) return error.message;
  }
  throw new Error("no RangeError");
}

describe("auth option checks", () => {
  const unsupported = (name: string) => `Unsupported provider '${name}'; choose one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`;
  const subscriptionOnly = "--login applies to subscription authentication";
  const azure = "https://x.openai.azure.com/";
  const refused = (name: string, reason: string) => `Invalid --header: header ${name} is refused: ${reason}`;

  /** Run `yian auth` on a store any read of which fails; expect exit 1 and `message` alone. */
  async function refuses(provider: string | undefined, options: Record<string, OptionValue>, message: string, env: Record<string, string> = {}) {
    const setup = setupAuth({ storeOpen: false });
    const io = fakeIo({ env: { ...setup.env, ...env } });
    const result = await run(auth, authArgs(provider, options), io, setup.deps);
    expect(result).toEqual({ code: 1, out: "", err: `${message}\n` });
    expect(result.err).not.toContain("private");
    expect(setup.service.calls).toEqual([]);
    expect(existsSync(setup.path)).toBe(false);
  }

  // Invalid setup options fail before external requests.
  test.each<[string | undefined, Record<string, OptionValue>, string]>([
    ["unknown", {}, unsupported("unknown")],
    ["openai", { method: "unknown" }, "Choose --method subscription or --method api-key"],
    ["openai", { method: "subscription", login: "unknown" }, "Choose --login browser or --login device"],
    ["openai", { method: "api-key", login: "device" }, subscriptionOnly],
  ])("invalid setup option: %p %j", async (provider, options, message) => {
    await refuses(provider, options, message);
  });

  // Option combinations fail before any request.
  test.each<[string | undefined, Record<string, OptionValue>, string]>([
    ["codex", {}, unsupported("codex")],
    ["codex", { default: true }, unsupported("codex")],
    ["anthropic", { method: "api-key" }, "--method and --login apply only to openai"],
    ["anthropic", { login: "device" }, "--method and --login apply only to openai"],
    ["openai", { endpoint: azure }, "--endpoint applies only to azure"],
    ["ollama", { "api-key-env": "X" }, "ollama takes no API key; omit --api-key-env"],
    ["anthropic", { "api-key-env": "" }, "--api-key-env needs an environment variable name"],
    ["openai", { remove: true, model: "x" }, "--remove does not take --model"],
    ["azure", { default: true, endpoint: azure }, "--default does not take --endpoint"],
  ])("option combination: %p %j", async (provider, options, message) => {
    await refuses(provider, options, message);
  });

  test("--default with --remove", async () => {
    await refuses("openai", { default: true, remove: true }, "Use either --default or --remove, not both");
  });

  // Azure options are checked before the store.
  test.each<[string | undefined, string, string]>([
    [undefined, "https://r.openai.azure.com/", "Azure needs OPENAI_API_VERSION set to its API version (runs read it too)"],
    [
      "2025-03-01-preview",
      "http://r.example.com",
      `Invalid Azure endpoint: ${rangeMessage(() => validateBaseUrl("http://r.example.com"))}`,
    ],
  ])("azure options: version %p, endpoint %p", async (version, endpoint, message) => {
    const env: Record<string, string> = version === undefined ? {} : { OPENAI_API_VERSION: version };
    await refuses("azure", { endpoint, model: "dep" }, message, env);
  });

  // Connection settings are refused before any request.
  const settingsEnv = { GW_TOKEN: "private-gw", EMPTY_TOKEN: "", BAD_TOKEN: " private-bad" };
  const subscription = { method: "subscription" };
  const apiKey = { method: "api-key" };
  test.each<[string | undefined, Record<string, OptionValue>, string]>([
    ["anthropic", { organization: "org-1" }, "--organization and --project apply only to openai"],
    ["anthropic", { project: "proj-1" }, "--organization and --project apply only to openai"],
    ["openai", { ...subscription, organization: "org-1" }, "--organization and --project apply to API-key authentication"],
    ["openai", { ...subscription, project: "proj-1" }, "--organization and --project apply to API-key authentication"],
    ["openai", { ...subscription, header: ["X-G=GW_TOKEN"] }, "--header applies to API-key authentication"],
    [
      "openai",
      { ...apiKey, organization: "org 1" },
      `Invalid --organization: ${rangeMessage(() => validateOrganization("org 1", "organization"))}`,
    ],
    ["ollama", { header: ["X-G=GW_TOKEN"] }, "ollama takes no API key, so it takes no --header"],
    ["anthropic", { header: ["Authorization=GW_TOKEN"] }, refused("Authorization", REFUSED_HEADER_NAMES["authorization"]!)],
    ["anthropic", { header: ["Content-Type=GW_TOKEN"] }, refused("Content-Type", REFUSED_HEADER_NAMES["content-type"]!)],
    [
      "openai",
      { ...apiKey, header: ["OpenAI-Project=GW_TOKEN"] },
      refused("OpenAI-Project", OPENAI_REFUSED_HEADER_NAMES["openai-project"]!),
    ],
    ["anthropic", { header: ["X G=GW_TOKEN"] }, `Invalid --header: ${rangeMessage(() => validateHeaderName("X G", "anthropic"))}`],
    ["anthropic", { header: ["X-G=UNSET_TOKEN"] }, "The environment variable given for --header X-G is not set"],
    ["anthropic", { header: ["X-G=EMPTY_TOKEN"] }, "The environment variable given for --header X-G is not set"],
    [
      "anthropic",
      { header: ["X-G=BAD_TOKEN"] },
      `Invalid --header: ${rangeMessage(() => validateHeaderValue("X-G", " private-bad"))}`,
    ],
    ["anthropic", { header: ["X-G="] }, "--header X-G=VARIABLE needs an environment variable name"],
    ["anthropic", { header: ["X-G=GW_TOKEN", "x-g=GW_TOKEN"] }, "--header x-g is given more than once (names ignore case)"],
    ["anthropic", { default: true, header: ["X-G=GW_TOKEN"] }, "--default does not take --header"],
    ["openai", { remove: true, organization: "org-1" }, "--remove does not take --organization"],
    [undefined, { project: "proj-1" }, "Name a provider: yian auth <provider> [options]"],
  ])("connection settings: %p %j", async (provider, options, message) => {
    await refuses(provider, options, message, settingsEnv);
  });

  test("an unset header variable is not echoed", async () => {
    await refuses(
      "anthropic",
      { header: ["X-G=sk_private_typed_value"], model: "m" },
      "The environment variable given for --header X-G is not set",
    );
  });

  test("a bare header without a terminal names the variable form", async () => {
    await refuses(
      "anthropic",
      { header: ["X-G"], model: "m1" },
      "--header X-G reads its value from a hidden prompt, which needs a terminal; use --header NAME=VARIABLE to read it from a variable",
      { ANTHROPIC_API_KEY: "private-key" },
    );
  });
});

/** Save an openai subscription, an anthropic key, an ollama login and an azure key with its endpoint. */
async function saveEachKind(store: AuthStore) {
  const oauth = new OAuthCredentials({
    accessToken: "private-access",
    refreshToken: "private-refresh",
    expiresAt: 9999999999,
    accountId: "test-account",
  });
  await store.save("openai", oauth, "gpt-test");
  await store.save("anthropic", new ApiKeyCredentials("private-key"), "claude-test");
  await store.save("ollama", null, "llama-test");
  await store.save("azure", new ApiKeyCredentials("private-azure"), "gpt-azure", { endpoint: "https://r.openai.azure.com/" });
}

/** The indented login lines of a listing, by provider. */
function loginLines(text: string): Record<string, string> {
  const lines = text.split("\n").filter((line) => line.startsWith("  "));
  return Object.fromEntries(lines.map((line) => [line.split(":")[0]!.trim(), line]));
}

describe("auth list, default and remove", () => {
  /** Run `yian auth [provider] options` on `setup`'s store with its environment changed by `env` (`null` unsets). */
  async function runAuth(
    setup: ReturnType<typeof setupAuth>,
    provider?: string,
    options: Record<string, OptionValue> = {},
    env: Record<string, string | null> = {},
  ) {
    const merged: Record<string, string> = { ...setup.env };
    for (const [name, value] of Object.entries(env)) {
      if (value === null) delete merged[name];
      else merged[name] = value;
    }
    const result = await run(auth, authArgs(provider, options), fakeIo({ env: merged }), setup.deps);
    return { ...result, output: result.out + result.err };
  }
  const noDefaults = { YIAN_DEFAULT_PROVIDER: null, YIAN_DEFAULT_MODEL: null };

  test("the list shows saved logins without secrets", async () => {
    const setup = setupAuth();
    let result = await runAuth(setup, undefined, {}, noDefaults);
    expect(result.code).toBe(0);
    expect(result.out).toBe("No login is saved; run 'yian auth <provider>' to add one\n");

    await saveEachKind(setup.store);
    await setup.store.setDefault("anthropic");
    result = await runAuth(setup, undefined, {}, noDefaults);
    expect(result.code).toBe(0);
    const lines = loginLines(result.out);
    expect(Object.keys(lines).sort()).toEqual(["anthropic", "azure", "ollama", "openai"]);
    expect(lines["openai"]).toBe("  openai: subscription, model gpt-test");
    expect(lines["anthropic"]).toBe("  anthropic: api-key, model claude-test (default)");
    expect(lines["ollama"]).toBe("  ollama: none, model llama-test");
    expect(lines["azure"]).toBe("  azure: api-key, model gpt-azure, endpoint https://r.openai.azure.com/");
    expect(result.output).not.toContain("private-");
    expect(result.output).not.toContain("YIAN_DEFAULT");
    expect(setup.service.calls).toEqual([]);

    result = await runAuth(setup, undefined, {}, { ...noDefaults, YIAN_DEFAULT_PROVIDER: "openai" });
    expect(result.code).toBe(0);
    expect(result.out).toContain("YIAN_DEFAULT_PROVIDER");
    expect(result.out).not.toContain("YIAN_DEFAULT_MODEL");
  });

  test("--default switches without sign-in", async () => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    for (const provider of ["openai", "ollama", "azure", "anthropic"]) {
      const result = await runAuth(setup, provider, { default: true });
      expect(result).toMatchObject({ code: 0, out: `Default provider: ${provider}\n` });
      expect(setup.store.read().defaultProvider).toBe(provider);
    }
    expect(setup.service.calls).toEqual([]);

    let result = await runAuth(setup, "google", { default: true });
    expect(result.code).toBe(1);
    expect(result.err).toBe("There is no saved google login; run 'yian auth' to list them\n");
    result = await runAuth(setup, "openai", { default: true, model: "x" });
    expect(result.code).toBe(1);
    expect(setup.store.read().defaultProvider).toBe("anthropic");
  });

  test("--remove deletes one login", async () => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    await setup.store.setDefault("openai");

    let result = await runAuth(setup, "anthropic", { remove: true }, noDefaults);
    expect(result).toMatchObject({ code: 0, out: "Removed the saved anthropic login\n" });
    let state = setup.store.read();
    expect(state.defaultProvider).toBe("openai");
    expect(state.selections.map((item) => item.provider).sort()).toEqual(["azure", "ollama", "openai"]);

    result = await runAuth(setup, "openai", { remove: true }, noDefaults);
    expect(result).toMatchObject({
      code: 0,
      out: "Removed the saved openai login\nNo default provider is saved; choose one with 'yian auth <provider> --default'\n",
    });
    expect(setup.store.read().defaultProvider).toBeNull();
    result = await runAuth(setup, undefined, {}, noDefaults);
    expect(result.code).toBe(0);
    expect(result.out).toContain("No default provider is saved; choose one with 'yian auth <provider> --default'\n");

    for (const provider of ["ollama", "azure"]) {
      expect((await runAuth(setup, provider, { remove: true })).code).toBe(0);
    }
    expect(setup.store.read().selections).toEqual([]);
    result = await runAuth(setup, "anthropic", { remove: true });
    expect(result.code).toBe(1);
    expect(result.err).toBe("There is no saved anthropic login; run 'yian auth' to list them\n");
    result = await runAuth(setup, "anthropic", { remove: true, default: true });
    expect(result.code).toBe(1);
    expect(setup.service.calls).toEqual([]);
  });

  test("--remove accepts an unsupported saved provider", async () => {
    const setup = setupAuth();
    const saved = {
      version: 1,
      default_provider: null,
      providers: {
        "legacy-x": { method: "api_key", model: "m", credential_id: "abcd", credentials: { api_key: "k" } },
      },
    };
    writeFileSync(setup.path, JSON.stringify(saved), { mode: 0o600 });
    chmodSync(setup.path, 0o600);
    const result = await runAuth(setup, "legacy-x", { remove: true });
    expect(result).toMatchObject({ code: 0, out: "Removed the saved legacy-x login\n" });
    expect(setup.store.read().selections).toEqual([]);
  });

  test("--remove finds a lowercase saved name", async () => {
    const setup = setupAuth();
    await setup.store.save("vllm", null, "m1");
    const result = await runAuth(setup, "VLLM", { remove: true });
    expect(result.code).toBe(0);
    expect(result.out).toStartWith("Removed the saved vllm login\n");
    expect(setup.store.read().selections).toEqual([]);
  });

  test("the environment override note agrees with the number of variables", async () => {
    const setup = setupAuth();
    let result = await runAuth(setup);
    expect(result.out).toBe("No login is saved; run 'yian auth <provider>' to add one\n");
    await setup.store.save("anthropic", new ApiKeyCredentials("private-key"), "claude-test");
    result = await runAuth(setup);
    expect(result.out).toEndWith(
      "YIAN_DEFAULT_PROVIDER and YIAN_DEFAULT_MODEL are set in the environment and override the saved choice\n",
    );
    result = await runAuth(setup, undefined, {}, { YIAN_DEFAULT_MODEL: null });
    expect(result.out).toEndWith("YIAN_DEFAULT_PROVIDER is set in the environment and overrides the saved choice\n");
  });

  test("the list shows connection settings by name only", async () => {
    const setup = setupAuth();
    await setup.store.save("openai", new ApiKeyCredentials("private-key", { "X-G": "private-gw" }), "gpt-test", {
      organization: "org-1",
      project: "proj-1",
    });
    await setup.store.save(
      "anthropic",
      new ApiKeyCredentials("private-key", { "X-B": "private-b", "X-A": "private-a" }),
      "claude-test",
    );
    const result = await runAuth(setup);
    expect(result.code).toBe(0);
    const lines = loginLines(result.out);
    expect(lines["openai"]).toBe("  openai: api-key, model gpt-test, organization org-1, project proj-1, headers X-G");
    expect(lines["anthropic"]).toBe("  anthropic: api-key, model claude-test, headers X-A, X-B (default)");
    expect(result.output).not.toContain("private-");
  });

  const jsonUsage = "Invalid value: --json applies to --remove and to openai --method subscription --login device";
  test.each<[string, string | undefined, Record<string, OptionValue>]>([
    ["the list", undefined, {}],
    ["--default", "openai", { default: true }],
    ["API-key setup", "openai", { method: "api-key", model: "gpt-test" }],
    ["browser sign-in", "openai", { method: "subscription", login: "browser" }],
  ])("--json on %s is a usage error", async (_, provider, options) => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    const before = readFileSync(setup.path);
    const result = await runAuth(setup, provider, { ...options, json: true });
    expect(result).toMatchObject({ code: 2, out: "", err: `Error: ${jsonUsage}\n` });
    expect(readFileSync(setup.path)).toEqual(before);
    expect(setup.service.calls).toEqual([]);
  });

  test("--remove --auth-file F without a provider deletes F", async () => {
    const setup = setupAuth();
    const path = join(tmp, "auth", "other.json");
    for (const json of [false, true]) {
      const store = new AuthStore(path);
      await saveEachKind(store);
      const result = await runAuth(setup, undefined, { remove: true, "auth-file": path, json });
      expect(result.code).toBe(0);
      if (json) expect(JSON.parse(result.out)).toEqual({ format: "yian.cli.v1", ok: true });
      else expect(result.out).toBe(`Removed the saved logins in ${path}\n`);
      expect(existsSync(path)).toBe(false);
    }
    expect(existsSync(setup.path)).toBe(false);
  });

  const wholeFile = "--remove without a provider deletes the whole --auth-file; it takes no --default or setup option";
  test.each<[string, Record<string, OptionValue>]>([
    ["--default", { default: true }],
    ["--model", { model: "x" }],
    ["--header", { header: ["X-G=GW_TOKEN"] }],
  ])("whole-file removal refuses %s", async (_, options) => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    const args = { remove: true, "auth-file": setup.path, ...options };
    const text = await runAuth(setup, undefined, args);
    expect(text).toMatchObject({ code: 1, out: "", err: `${wholeFile}\n` });
    const json = await runAuth(setup, undefined, { ...args, json: true });
    expect(json.code).toBe(1);
    expect(JSON.parse(json.out)).toEqual({ format: "yian.cli.v1", error: { code: "auth.failed", message: wholeFile } });
    expect(existsSync(setup.path)).toBe(true);
  });

  test("--remove without a provider or --auth-file names a provider", async () => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    const message = "Name a provider: yian auth <provider> [options]";
    const text = await runAuth(setup, undefined, { remove: true });
    expect(text).toMatchObject({ code: 1, out: "", err: `${message}\n` });
    const json = await runAuth(setup, undefined, { remove: true, json: true });
    expect(json.code).toBe(1);
    expect(JSON.parse(json.out)).toEqual({ format: "yian.cli.v1", error: { code: "auth.failed", message } });
    expect(existsSync(setup.path)).toBe(true);
  });

  test("a JSON removal prints ok, or an auth.failed document", async () => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    const removed = await runAuth(setup, "anthropic", { remove: true, json: true });
    expect(removed.code).toBe(0);
    expect(removed.out).toBe('{"format":"yian.cli.v1","ok":true}\n');
    const missing = await runAuth(setup, "anthropic", { remove: true, json: true });
    expect(missing).toMatchObject({ code: 1, err: "" });
    expect(JSON.parse(missing.out)).toEqual({
      format: "yian.cli.v1",
      error: { code: "auth.failed", message: "There is no saved anthropic login; run 'yian auth' to list them" },
    });
  });

  test.each([false, true])("SIGINT during a removal waiting for the lock stops it (--json %p)", async (json) => {
    const setup = setupAuth();
    await saveEachKind(setup.store);
    const signals = new EventEmitter();
    const holder = lockFile(`${setup.path}.lock`);
    // The store's first try at the lock fails busy and goes on to sleep, all in the same turn of the
    // event loop as this lstat; so once `tried` resolves, the removal is waiting for the lock.
    const lstat = fs.lstatSync;
    let tried!: () => void;
    const trying = new Promise<void>((resolve) => (tried = resolve));
    const spy = spyOn(fs, "lstatSync").mockImplementation(((path: fs.PathLike, ...rest: unknown[]) => {
      if (path === `${setup.path}.lock`) tried();
      return (lstat as (...args: unknown[]) => unknown)(path, ...rest);
    }) as typeof fs.lstatSync);
    try {
      const io = fakeIo({ env: setup.env });
      const running = auth(authArgs("anthropic", { remove: true, json }), io.io, { ...setup.deps, signals });
      await trying;
      expect(signals.listenerCount("SIGINT")).toBe(1);
      expect(signals.listenerCount("SIGTERM")).toBe(0);
      signals.emit("SIGINT");
      const code = await running;
      if (json) expect({ code, out: io.out(), err: io.err() }).toEqual({ code: 130, out: "", err: "" });
      else expect({ code, out: io.out(), err: io.err() }).toEqual({ code: 1, out: "", err: `${CANCELLED}\n` });
    } finally {
      spy.mockRestore();
      holder.release();
    }
    expect(signals.listenerCount("SIGINT")).toBe(0);
    expect(signals.listenerCount("SIGTERM")).toBe(0);
    expect(setup.store.selection("anthropic")?.model).toBe("claude-test");
  });

  test("a finished command leaves no signal listener", async () => {
    const setup = setupAuth();
    const signals = new EventEmitter();
    const result = await run(auth, authArgs(), fakeIo({ env: setup.env }), { ...setup.deps, signals });
    expect(result.code).toBe(0);
    expect(signals.listenerCount("SIGINT")).toBe(0);
  });
});

describe("auth device sign-in as JSON lines", () => {
  const DEVICE_JSON = { method: "subscription", login: "device", json: true };
  const CODE = {
    format: "yian.cli.v1",
    event: "code",
    user_code: "TEST-CODE",
    verification_uri: "https://example.invalid/device",
  };
  const WAITING = { format: "yian.cli.v1", event: "waiting" };

  /**
   * `setupAuth` with an emitter as the signal source and a service factory
   * recording the options each service is built with.
   */
  function deviceJson() {
    const setup = setupAuth();
    const signals = new EventEmitter();
    const service = setup.service.service;
    const built: { signal?: AbortSignal }[] = [];
    const deps: AuthDeps = {
      openAIAuth: (options) => {
        built.push(options);
        return service;
      },
      signals,
    };
    return { ...setup, service, calls: setup.service.calls, signals, built, deps };
  }

  /** Run `yian auth openai --method subscription --login device --json` with `options`; parse each stdout line. */
  async function runDevice(setup: ReturnType<typeof deviceJson>, options: Record<string, OptionValue> = {}) {
    const io = fakeIo({ env: setup.env });
    const result = await run(auth, authArgs("openai", { ...DEVICE_JSON, ...options }), io, setup.deps);
    return { ...result, lines: result.out.split("\n").filter(Boolean).map((line) => JSON.parse(line)) };
  }

  /** How many SIGINT and SIGTERM listeners `signals` has. */
  const listeners = (signals: EventEmitter) => signals.listenerCount("SIGINT") + signals.listenerCount("SIGTERM");

  // Device json stopped before the save exits 130 and saves nothing.
  test.each(["SIGTERM", "SIGINT"] as const)("%s before the save exits 130 and saves nothing", async (name) => {
    const setup = deviceJson();
    const counts: number[] = [];
    setup.service.subscriptionModels = async () => {
      counts.push(listeners(setup.signals));
      setup.signals.emit(name);
      counts.push(listeners(setup.signals));
      // No listener takes a second signal, so a process gets its default action and ends.
      setup.signals.emit(name);
      return ["gpt-test"];
    };
    const result = await runDevice(setup);
    expect(result).toMatchObject({ code: 130, err: "" });
    expect(result.lines).toEqual([CODE, WAITING]);
    expect(existsSync(setup.path)).toBe(false);
    expect(counts).toEqual([2, 0]);
    expect(listeners(setup.signals)).toBe(0);
    expect(setup.built).toHaveLength(1);
    expect(setup.built[0]!.signal?.aborted).toBe(true);
  });

  // Device json stopped after the rename is saved and done.
  test("SIGTERM after the rename is saved and done", async () => {
    const setup = deviceJson();
    const rename = fs.renameSync;
    let renames = 0;
    const spy = spyOn(fs, "renameSync").mockImplementation((...args) => {
      rename(...args);
      renames += 1;
      setup.signals.emit("SIGTERM");
    });
    let result: Awaited<ReturnType<typeof runDevice>>;
    try {
      result = await runDevice(setup);
    } finally {
      spy.mockRestore();
    }
    expect(renames).toBe(1);
    expect(result).toMatchObject({ code: 0, err: "" });
    expect(result.lines).toEqual([
      CODE,
      WAITING,
      { format: "yian.cli.v1", event: "done", models: ["gpt-test", "gpt-other"], model: "gpt-test" },
    ]);
    expect(result.out).not.toContain("private-");
    expect(setup.store.read().defaultProvider).toBe("openai");
    expect(setup.store.selection()?.model).toBe("gpt-test");
    expect(setup.calls).toEqual(["device", "subscription_models"]);
    expect(listeners(setup.signals)).toBe(0);
    setup.signals.emit("SIGTERM");
    expect(listeners(setup.signals)).toBe(0);
  });

  test("a --model the subscription doesn't offer is an error line, and nothing is saved", async () => {
    const setup = deviceJson();
    const result = await runDevice(setup, { model: "unavailable" });
    expect(result).toMatchObject({ code: 1, err: "" });
    expect(result.lines).toEqual([
      CODE,
      WAITING,
      {
        format: "yian.cli.v1",
        event: "error",
        error: { code: "auth.failed", message: "The selected model is not available for this connection" },
      },
    ]);
    expect(existsSync(setup.path)).toBe(false);
    expect(listeners(setup.signals)).toBe(0);
  });

  test("a malformed store is one error line before any service call", async () => {
    const setup = deviceJson();
    writeFileSync(setup.path, '{"version": 1, "providers": {}}', { mode: 0o600 });
    chmodSync(setup.path, 0o600);
    const before = readFileSync(setup.path);
    let message = "";
    try {
      setup.store.read();
    } catch (error) {
      message = (error as AuthError).message;
    }
    expect(message).not.toBe("");
    const result = await runDevice(setup);
    expect(result).toMatchObject({ code: 1, err: "" });
    expect(result.lines).toEqual([{ format: "yian.cli.v1", event: "error", error: { code: "auth.failed", message } }]);
    expect(setup.built).toEqual([]);
    expect(setup.calls).toEqual([]);
    expect(readFileSync(setup.path)).toEqual(before);
  });

  test("the login becomes the default only when the store has none", async () => {
    const setup = deviceJson();
    await setup.store.save("anthropic", new ApiKeyCredentials("private-key"), "claude-test");
    const result = await runDevice(setup, { model: "gpt-other" });
    expect(result.code).toBe(0);
    expect(result.lines.at(-1)).toEqual({
      format: "yian.cli.v1",
      event: "done",
      models: ["gpt-test", "gpt-other"],
      model: "gpt-other",
    });
    expect(setup.store.read().defaultProvider).toBe("anthropic");
    expect(setup.store.selection("openai")?.model).toBe("gpt-other");
  });
});

/** An answer the fake terminal never gives: the prompt waits until its signal aborts, then rejects with its reason. */
const WAIT = Symbol("wait");

/**
 * A fake prompt terminal recording each prompt (kind, label, choices, initial, secret) and
 * answering from `answers` in order (`null` = cancel, `WAIT` = wait for the signal). It counts the
 * `close()` calls.
 */
function fakeTerminal(answers: (string | null | typeof WAIT)[]) {
  const asked: { kind: string; label: string; choices?: readonly string[]; initial?: string; secret?: boolean }[] = [];
  const signals: (AbortSignal | undefined)[] = [];
  let closed = 0;
  const answer = (signal: AbortSignal | undefined): Promise<string | null> => {
    signals.push(signal);
    if (answers.length === 0) throw new Error("unexpected prompt");
    const next = answers.shift()!;
    if (next !== WAIT) return Promise.resolve(next);
    return new Promise((_, reject) => signal?.addEventListener("abort", () => reject(signal.reason), { once: true }));
  };
  const terminal = {
    askText: async (label: string, options: { initial?: string; secret?: boolean; signal?: AbortSignal } = {}) => {
      asked.push({ kind: "text", label, secret: options.secret === true });
      return answer(options.signal);
    },
    askChoice: async (label: string, choices: readonly string[], options: { initial?: string; signal?: AbortSignal } = {}) => {
      asked.push({ kind: "choice", label, choices: [...choices], initial: options.initial });
      return answer(options.signal);
    },
    close: () => void (closed += 1),
  };
  return { terminal, asked, signals, closed: () => closed };
}

describe("auth OpenAI setup", () => {
  /** Run `yian auth openai options` on `setup`'s store, with `env` added and `deps` overriding. */
  async function runSetup(
    setup: ReturnType<typeof setupAuth>,
    options: Record<string, OptionValue> = {},
    { env = {}, deps = {}, isTTY }: { env?: Record<string, string>; deps?: AuthDeps; isTTY?: boolean } = {},
  ) {
    const io = fakeIo({ env: { ...setup.env, ...env }, isTTY });
    const result = await run(auth, authArgs("openai", options), io, { ...setup.deps, ...deps });
    return { ...result, output: result.out + result.err };
  }

  const METHOD = { kind: "choice", label: "OpenAI authentication method:", choices: ["subscription", "api-key"] };
  const LOGIN = { kind: "choice", label: "Subscription sign-in:", choices: ["browser", "device"] };

  // Explicit setup saves method and model without echoing secrets.
  test.each<[string, string | undefined, string]>([
    ["subscription", "device", "Open https://example.invalid/device and enter code: TEST-CODE\n"],
    ["subscription", "browser", ""],
    ["api-key", undefined, ""],
  ])("explicit setup: --method %p --login %p", async (method, login, shown) => {
    const setup = setupAuth();
    const options: Record<string, OptionValue> = { method, model: "gpt-test" };
    if (login) options["login"] = login;
    const result = await runSetup(setup, options);
    expect(result).toEqual({
      code: 0,
      out: `${shown}Saved OpenAI ${method} authentication with model gpt-test\nDefault provider: openai\n`,
      err: "",
      output: expect.any(String),
    });
    const selected = setup.store.selection()!;
    expect([selected.provider, selected.model]).toEqual(["openai", "gpt-test"]);
    expect(selected.method).toBe(method === "api-key" ? "api_key" : "subscription");
    expect(result.output).not.toContain("private-");
    expect(setup.service.calls).toEqual(login ? [login, "subscription_models"] : ["api_models"]);
    expect(setup.opened).toEqual(login === "browser" ? ["https://example.invalid/authorize"] : []);
  });

  test.each<[string, boolean, string]>([
    ["stripped on a TTY", true, "Open https://example.invalid/device and enter code: TEST-CODE\n"],
    ["kept exactly when piped", false, "Open https://example.invalid/device and enter code: TEST\x1b]52;c;aGk=\x07-CODE\n"],
  ])("the device sign-in's code is %s", async (_, outTty, shown) => {
    const setup = setupAuth();
    const deviceLogin = setup.service.service.deviceLogin;
    setup.service.service.deviceLogin = (display) =>
      deviceLogin((url, code) => display(url, code.replace("-", "\x1b]52;c;aGk=\x07-")));
    const io = fakeIo({ env: setup.env, outTty });
    const result = await run(auth, authArgs("openai", { method: "subscription", login: "device", model: "gpt-test" }), io, setup.deps);
    expect(result.code).toBe(0);
    expect(result.out).toStartWith(shown);
  });

  // Interactive setup defaults to subscription and catalog first model.
  test("interactive setup defaults to a subscription and the catalog's first model", async () => {
    const setup = setupAuth();
    const fake = fakeTerminal(["subscription", "device", "gpt-test"]);
    const result = await runSetup(setup, {}, { deps: { terminal: fake.terminal } });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([
      { ...METHOD, initial: "subscription" },
      { ...LOGIN, initial: "browser" },
      { kind: "choice", label: "Model:", choices: ["gpt-test", "gpt-other"], initial: "gpt-test" },
    ]);
    expect(fake.signals.every((signal) => signal instanceof AbortSignal && !signal.aborted)).toBe(true);
    // The service gets the stop the prompts get.
    expect(setup.service.built).toHaveLength(1);
    expect(setup.service.built[0]!.signal).toBe(fake.signals[0]!);
    expect(fake.closed()).toBe(1);
    expect(setup.store.selection()?.model).toBe("gpt-test");
  });

  test("the sign-in picker starts at device over SSH", async () => {
    const setup = setupAuth();
    const fake = fakeTerminal(["device"]);
    const env = { SSH_CONNECTION: "10.0.0.1 22 10.0.0.2 22" };
    const result = await runSetup(setup, { method: "subscription", model: "gpt-test" }, { env, deps: { terminal: fake.terminal } });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([{ ...LOGIN, initial: "device" }]);
  });

  // Failed or cancelled setup preserves previous connection.
  test.each(["login", "model", "cancel", "missing_model"])("a %s failure keeps the previous login", async (failure) => {
    const setup = setupAuth();
    const previous = await setup.store.save("openai", new ApiKeyCredentials("old-key"), "old-model");
    const before = readFileSync(setup.path);
    const fail = async (): Promise<never> => {
      throw new AuthError("Simulated setup failure");
    };
    if (failure === "login") setup.service.service.deviceLogin = fail;
    if (failure === "model") setup.service.service.subscriptionModels = fail;
    const options: Record<string, OptionValue> = { method: "subscription", login: "device" };
    if (failure !== "cancel") options["model"] = failure === "missing_model" ? "unavailable" : "gpt-test";
    const fake = fakeTerminal([null]);
    const result = await runSetup(setup, options, { deps: { terminal: fake.terminal } });
    expect(result.code).toBe(1);
    const expected = {
      login: "Simulated setup failure",
      model: "Simulated setup failure",
      cancel: CANCELLED,
      missing_model: "The selected model is not available for this connection",
    }[failure];
    expect(result.err).toBe(`${expected}\n`);
    expect(readFileSync(setup.path)).toEqual(before);
    expect(setup.store.selection()).toEqual(previous);
    expect(result.output).not.toContain("old-key");
    expect(result.output).not.toContain("private-");
  });

  // Noninteractive setup requires explicit choices.
  test.each<[string, boolean, AuthDeps]>([
    ["stdin is not a TTY", false, {}],
    ["the terminal is null", true, { terminal: null }],
  ])("non-interactive setup requires explicit choices (%s)", async (_, isTTY, deps) => {
    const setup = setupAuth();
    const result = await runSetup(setup, {}, { isTTY, deps });
    expect(result).toMatchObject({
      code: 1,
      out: "",
      err: "Authentication needs --method when no interactive terminal is available\n",
    });
    expect(setup.service.calls).toEqual([]);
    expect(existsSync(setup.path)).toBe(false);
  });

  test("a TTY stdin without deps.terminal prompts on a real terminal, which reads EOF as a cancel", async () => {
    const setup = setupAuth();
    // Swallowed, so the prompt doesn't draw on the test's terminal; the fake stdin has no raw mode.
    const drawn = spyOn(process.stderr, "write").mockImplementation(() => true);
    let result: Awaited<ReturnType<typeof runSetup>>;
    let draws: number;
    try {
      result = await runSetup(setup, {}, { isTTY: true });
      draws = drawn.mock.calls.length;
    } finally {
      drawn.mockRestore();
    }
    expect(result).toMatchObject({ code: 1, out: "", err: `${CANCELLED}\n` });
    expect(draws).toBeGreaterThan(0);
    expect(setup.service.calls).toEqual([]);
    expect(existsSync(setup.path)).toBe(false);
  });

  // Setup keeps existing default.
  test("setup keeps an existing default", async () => {
    const setup = setupAuth();
    const options = { method: "api-key", model: "gpt-test" };
    let result = await runSetup(setup, options);
    expect(result.code).toBe(0);
    expect(setup.store.read().defaultProvider).toBe("openai");
    expect(result.output).toContain("Default provider: openai");

    result = await runSetup(setup, options);
    expect(result.code).toBe(0);
    expect(result.output).toContain("Default provider: openai");
    expect(result.output).not.toContain("switch with");

    await setup.store.save("anthropic", new ApiKeyCredentials("private-other"), "claude-test");
    result = await runSetup(setup, options);
    expect(result.code).toBe(0);
    expect(setup.store.read().defaultProvider).toBe("anthropic");
    expect(setup.store.selection("openai")?.model).toBe("gpt-test");
    expect(result.out).toEndWith("Default provider: anthropic (switch with 'yian auth openai --default')\n");
  });

  // Openai api key setup saves connection settings.
  test("API-key setup saves the connection settings", async () => {
    const setup = setupAuth();
    const result = await runSetup(
      setup,
      { method: "api-key", model: "gpt-test", organization: "org-1", project: "proj-1", header: ["X-G=GW_TOKEN"] },
      { env: { GW_TOKEN: "private-gw" } },
    );
    expect(result.code).toBe(0);
    expect(setup.service.settings).toEqual([
      { apiKey: "private-key", headers: [["X-G", "private-gw"]], organization: "org-1", project: "proj-1" },
    ]);
    const selected = setup.store.selection("openai")!;
    expect([selected.organization, selected.project]).toEqual(["org-1", "proj-1"]);
    expect(selected.headerNames).toEqual(["X-G"]);
    const credentials = (await setup.store.credentials("openai", selected.credentialId)) as ApiKeyCredentials;
    expect(credentials.headers).toEqual([["X-G", "private-gw"]]);
    expect(result.output).not.toContain("private-");
  });

  test("API-key setup reads the key and a bare header's value from hidden prompts", async () => {
    const setup = setupAuth();
    const { OPENAI_API_KEY: _, ...env } = setup.env;
    setup.env = env;
    const fake = fakeTerminal(["  typed-key  ", "typed-gw"]);
    const options = { method: "api-key", model: "gpt-test", header: ["X-G"] };
    const result = await runSetup(setup, options, { deps: { terminal: fake.terminal } });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([
      { kind: "text", label: "openai API key:", secret: true },
      { kind: "text", label: "X-G header value:", secret: true },
    ]);
    expect(setup.service.settings).toEqual([{ apiKey: "typed-key", headers: [["X-G", "typed-gw"]] }]);
    expect(result.output).not.toContain("typed-");
  });

  /** An `openAIAuth` building the real service over a fake transport that records each request. */
  function recordedService(models: readonly string[]) {
    const sent: { url: string; headers: Record<string, string> }[] = [];
    const request: AuthRequest = async (_method, url, options) => {
      sent.push({ url, headers: options.headers });
      return new HttpResponse(200, { data: models.map((id) => ({ id })) });
    };
    const openAIAuth: AuthDeps["openAIAuth"] = (options) => new OpenAIAuth({ ...options, request });
    return { sent, openAIAuth };
  }

  test("API-key setup lists models at OPENAI_BASE_URL and sends nothing to api.openai.com", async () => {
    const setup = setupAuth();
    const recorded = recordedService(["gateway-model"]);
    const result = await runSetup(
      setup,
      { method: "api-key", model: "gateway-model", header: ["X-G=GW_TOKEN"] },
      {
        env: { OPENAI_BASE_URL: "https://gateway.example.com/v1/", GW_TOKEN: "private-gw" },
        deps: { openAIAuth: recorded.openAIAuth },
      },
    );
    expect(result.code).toBe(0);
    expect(recorded.sent.map((request) => request.url)).toEqual(["https://gateway.example.com/v1/models"]);
    expect(recorded.sent[0]!.headers).toMatchObject({ Authorization: "Bearer private-key", "X-G": "private-gw" });
    expect(recorded.sent.some((request) => request.url.includes("api.openai.com"))).toBe(false);
    expect(setup.store.selection("openai")?.model).toBe("gateway-model");
    expect(result.output).not.toContain("private-");
  });

  test("API-key setup refuses an invalid OPENAI_BASE_URL before sending the key anywhere", async () => {
    const setup = setupAuth();
    const recorded = recordedService(["gpt-test"]);
    const result = await runSetup(
      setup,
      { method: "api-key", model: "gpt-test" },
      {
        env: { OPENAI_BASE_URL: "https://user:private-pw@gateway.example.com/v1" },
        deps: { openAIAuth: recorded.openAIAuth },
      },
    );
    expect(result.code).toBe(1);
    expect(result.err).toStartWith("Invalid OPENAI_BASE_URL: base_url must be");
    expect(recorded.sent).toEqual([]);
    expect(result.output).not.toContain("private-");
    expect(existsSync(setup.path)).toBe(false);
  });

  test("API-key setup with an empty OPENAI_BASE_URL lists OpenAI's models, as runs use OpenAI's host", async () => {
    const setup = setupAuth();
    const recorded = recordedService(["gpt-test"]);
    const result = await runSetup(
      setup,
      { method: "api-key", model: "gpt-test" },
      { env: { OPENAI_BASE_URL: "" }, deps: { openAIAuth: recorded.openAIAuth } },
    );
    expect(result.code).toBe(0);
    expect(recorded.sent.map((request) => request.url)).toEqual(["https://api.openai.com/v1/models"]);
  });

  test.each<[string, (string | null)[], string]>([
    ["a cancelled key prompt", [null], CANCELLED],
    ["an empty key", ["  "], "Authentication cancelled; no API key was supplied"],
    ["a cancelled header prompt", ["typed-key", null], CANCELLED],
    ["an empty header value", ["typed-key", ""], "Authentication cancelled; no value was given for X-G"],
  ])("%s saves nothing", async (_, answers, message) => {
    const setup = setupAuth();
    const { OPENAI_API_KEY: __, ...env } = setup.env;
    setup.env = env;
    const fake = fakeTerminal(answers);
    const options = { method: "api-key", model: "gpt-test", header: ["X-G"] };
    const result = await runSetup(setup, options, { deps: { terminal: fake.terminal } });
    expect(result).toMatchObject({ code: 1, err: `${message}\n` });
    expect(setup.service.calls).toEqual([]);
    expect(existsSync(setup.path)).toBe(false);
  });

  test("API-key setup without the key or a terminal names the variable", async () => {
    const setup = setupAuth();
    const result = await runSetup(setup, { method: "api-key", model: "gpt-test", "api-key-env": "MY_KEY" });
    expect(result).toMatchObject({ code: 1, err: "Set MY_KEY before API-key setup, or run in an interactive terminal\n" });
    expect(existsSync(setup.path)).toBe(false);
  });

  // Subscription chosen at the prompt refuses settings.
  test("a subscription chosen at the prompt refuses settings", async () => {
    const setup = setupAuth();
    const fake = fakeTerminal(["subscription"]);
    const result = await runSetup(setup, { organization: "org-1" }, { deps: { terminal: fake.terminal } });
    expect(result).toMatchObject({
      code: 1,
      err: "--organization, --project and --header apply to API-key authentication\n",
    });
    expect(setup.service.calls).toEqual([]);
    expect(existsSync(setup.path)).toBe(false);
  });

  // Openai listing failure is scrubbed of header values.
  test("the listing failure is scrubbed of header values", async () => {
    const setup = setupAuth();
    setup.service.service.apiModels = async () => {
      throw new AuthError("rejected gwvalue7 with private-key");
    };
    const result = await runSetup(
      setup,
      { method: "api-key", model: "gpt-test", header: ["X-G=GW_TOKEN"] },
      { env: { GW_TOKEN: "gwvalue7" } },
    );
    expect(result).toMatchObject({ code: 1, err: "rejected [REDACTED] with [REDACTED]\n" });
    expect(existsSync(setup.path)).toBe(false);
  });

  test("no text model in the listing needs --model", async () => {
    const setup = setupAuth();
    setup.service.service.apiModels = async () => ["embedding-test", "gpt-realtime-test"];
    const fake = fakeTerminal([]);
    const result = await runSetup(setup, { method: "api-key" }, { deps: { terminal: fake.terminal } });
    expect(result).toMatchObject({
      code: 1,
      err: "No general text models were returned; use --model with an available text model\n",
    });
  });

  test("a SIGINT while a picker waits cancels the setup and saves nothing", async () => {
    const setup = setupAuth();
    const signals = new EventEmitter();
    const fake = fakeTerminal([WAIT]);
    const io = fakeIo({ env: setup.env });
    const running = auth(authArgs("openai"), io.io, { ...setup.deps, terminal: fake.terminal, signals });
    while (fake.asked.length === 0) await new Promise((resolve) => setTimeout(resolve, 1));
    signals.emit("SIGINT");
    expect({ code: await running, out: io.out(), err: io.err() }).toEqual({ code: 1, out: "", err: `${CANCELLED}\n` });
    expect(fake.closed()).toBe(1);
    expect(signals.listenerCount("SIGINT")).toBe(0);
    expect(setup.service.calls).toEqual([]);
    expect(existsSync(setup.path)).toBe(false);
  });
});

/** An error named `name`. */
function named(name: string, message: string): Error {
  return Object.assign(new Error(message), { name });
}

/**
 * A fake `chatModel` recording each config and options. Its model's `stream` records each request
 * and the signal it got, then throws `failure` when set, waits for its signal when `hang` is set
 * (rejecting with the signal's reason), or replies "OK". `build` runs with the options before the
 * model is returned.
 */
function fakeChatModel() {
  const fake = {
    built: [] as { config: Record<string, unknown>; options: Parameters<typeof chatModelFromConfig>[1] & {} }[],
    requests: [] as Request[],
    signals: [] as (AbortSignal | undefined)[],
    failure: undefined as unknown,
    hang: false,
    build: async (_config: Record<string, unknown>, _options: Parameters<typeof chatModelFromConfig>[1] & {}) => {},
    chatModel: undefined as unknown as NonNullable<AuthDeps["chatModel"]>,
  };
  async function* reply(signal: AbortSignal | undefined): AsyncGenerator<Event> {
    if (fake.failure !== undefined) throw fake.failure;
    if (fake.hang) {
      await new Promise((_, reject) => signal?.addEventListener("abort", () => reject(signal.reason), { once: true }));
    }
    yield new TextDelta(0, "OK");
    yield new PartDone(0, new TextPart("OK"));
    yield new Finish({ reason: "stop" });
  }
  const model = {
    provider: "fake",
    model: "fake",
    stream: (request: Request, options: { signal?: AbortSignal } = {}) => {
      fake.requests.push(request);
      fake.signals.push(options.signal);
      return reply(options.signal);
    },
  };
  fake.chatModel = (async (config: Record<string, unknown>, options = {}) => {
    fake.built.push({ config: { ...config }, options });
    await fake.build(config, options);
    return model;
  }) as unknown as NonNullable<AuthDeps["chatModel"]>;
  return fake;
}

describe("modelAnswers", () => {
  const never = new AbortController().signal;
  const io = (env: Record<string, string> = {}) => fakeIo({ env }).io;

  // Model answers builds and invokes once.
  test("builds the model from what it is given and sends it one prompt", async () => {
    const fake = fakeChatModel();
    const deps = { chatModel: fake.chatModel };
    const azureIo = io();
    expect(await modelAnswers("azure", "dep", "private-key", "https://r/", {}, azureIo, deps, never)).toBeNull();
    const [{ config, options }] = fake.built as [(typeof fake.built)[0]];
    expect(config).toEqual({ provider: "azure", model: "dep", api_key: "private-key", base_url: "https://r/" });
    expect(options.env).toBe(azureIo.env);
    expect(options.snapshot).toBeInstanceOf(ConnectionSnapshot);
    expect([options.snapshot!.defaults, options.snapshot!.providerModels, options.snapshot!.connections]).toEqual([
      {},
      {},
      {},
    ]);
    expect(options.signal).toBeInstanceOf(AbortSignal);
    expect(options.onWarning).toBeInstanceOf(Function);
    expect(fake.requests).toEqual([
      new Request({ messages: [new Message({ role: "user", parts: [new TextPart("Reply with OK.")] })] }),
    ]);
    expect(fake.signals).toEqual([options.signal]);

    fake.failure = named("RuntimeError", "bad key private-key");
    expect(await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never)).toBe(
      "RuntimeError: bad key [REDACTED]",
    );
    fake.failure = named("Echo", "rejected\n  input private-key");
    expect(await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never)).toBe(
      "Echo: rejected input [REDACTED]",
    );

    fake.failure = undefined;
    // A keyless vLLM login is checked with the client's dummy key, not VLLM_API_KEY.
    const ambient = io({ VLLM_API_KEY: "private-ambient" });
    expect(await modelAnswers("vllm", "m1", null, null, {}, ambient, deps, never)).toBeNull();
    expect(fake.built.at(-1)!.config).toEqual({ provider: "vllm", model: "m1", api_key: "EMPTY" });
    expect(await modelAnswers("ollama", "m1", null, null, {}, ambient, deps, never)).toBeNull();
    expect(fake.built.at(-1)!.config).toEqual({ provider: "ollama", model: "m1" });
  });

  // Model answers mutes the unknown model warning.
  test("mutes only the unknown-model warning", async () => {
    const emitted = spyOn(process, "emitWarning").mockImplementation(() => {});
    try {
      const fake = fakeChatModel();
      const other = `Another warning ${Math.random()}`;
      const received: string[] = [];
      fake.build = async (config, options) => {
        const onWarning = (message: string) => {
          received.push(message);
          options.onWarning!(message);
        };
        // The real factory, for its warning text; the build sends nothing.
        await chatModelFromConfig(config, { ...options, onWarning });
        onWarning(other);
      };
      const deps = { chatModel: fake.chatModel };
      expect(await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never)).toBeNull();
      expect(received).toEqual([
        "Model 'claude-x' is not in the known model list for provider 'anthropic'. Continuing anyway.",
        other,
      ]);
      expect(emitted.mock.calls as unknown[][]).toEqual([[other, "RuntimeWarning"]]);
    } finally {
      emitted.mockRestore();
    }
  });

  // Model answers scrubs the azure ad token.
  test("scrubs the Azure AD token", async () => {
    const [token, key] = ["private-ad-token-k7qz", "private-azure-key-m2vd"];
    const fake = fakeChatModel();
    fake.failure = named("RuntimeError", `bad token ${token} key ${key}`);
    const env = { YIAN_AUTH_FILE: join(tmp, "auth.json"), AZURE_OPENAI_AD_TOKEN: token };
    const deps = { chatModel: fake.chatModel };
    const problem = await modelAnswers("azure", "dep", key, "https://r.openai.azure.com/", {}, io(env), deps, never);
    expect(problem).toBe("RuntimeError: bad token [REDACTED] key [REDACTED]");
  });

  // Model answers sends and scrubs connection headers.
  test("sends the connection headers and scrubs their values", async () => {
    const fake = fakeChatModel();
    fake.failure = named("RuntimeError", "gateway rejected gwvalue7");
    const deps = { chatModel: fake.chatModel };
    const problem = await modelAnswers("anthropic", "claude-x", "private-key", null, { "X-G": "gwvalue7" }, io(), deps, never);
    expect(fake.built[0]!.config["headers"]).toEqual({ "X-G": "gwvalue7" });
    expect(problem).toBe("RuntimeError: gateway rejected [REDACTED]");

    fake.failure = undefined;
    expect(await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never)).toBeNull();
    expect(fake.built.at(-1)!.config).not.toHaveProperty("headers");
  });

  test("a failure folds whitespace and is cut at 200 characters after the name", async () => {
    const fake = fakeChatModel();
    fake.failure = named("ProviderError", `${"a".repeat(150)}\n\t  ${"b".repeat(100)} `);
    const problem = await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), { chatModel: fake.chatModel }, never);
    expect(problem).toBe(`ProviderError: ${`${"a".repeat(150)} ${"b".repeat(100)}`.slice(0, 200)}`);
  });

  test("a thrown non-Error shows as itself, redacted and folded", async () => {
    const fake = fakeChatModel();
    fake.failure = "refused  private-key\n";
    const problem = await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), { chatModel: fake.chatModel }, never);
    expect(problem).toBe("refused [REDACTED]");
  });

  test("a model that doesn't reply within 30 seconds times out", async () => {
    const fake = fakeChatModel();
    fake.hang = true;
    const scheduler = virtualScheduler();
    const deps = { chatModel: fake.chatModel, scheduler };
    const problem = await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never);
    expect(problem).toBe("TimeoutError: The model didn't reply within 30 seconds");
    expect(scheduler.now()).toBe(30_000);
  });

  test("any failure once the deadline passed reads as the timeout", async () => {
    const fake = fakeChatModel();
    const scheduler = virtualScheduler();
    // A build cut short by the deadline that rejects with its own error, not the signal's reason.
    fake.build = (_, options) =>
      new Promise((_, reject) =>
        options.signal!.addEventListener("abort", () => reject(new Error("aborted")), { once: true }),
      );
    const deps = { chatModel: fake.chatModel, scheduler };
    const problem = await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never);
    expect(problem).toBe("TimeoutError: The model didn't reply within 30 seconds");
  });

  test("no deadline sleeper outlives a check", async () => {
    const fake = fakeChatModel();
    const scheduler = virtualScheduler();
    const deps = { chatModel: fake.chatModel, scheduler };
    expect(await modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), deps, never)).toBeNull();
    // A sleeper left behind would move the virtual clock to the 30 s deadline meanwhile.
    await new Promise((resolve) => setTimeout(resolve, 10));
    await scheduler.sleep(1);
    expect(scheduler.now()).toBe(1);
  });

  test("a stop during the check rejects with the stop's reason", async () => {
    const fake = fakeChatModel();
    fake.hang = true;
    const stop = new AbortController();
    const reason = new Error("stopped");
    const checking = modelAnswers("anthropic", "claude-x", "private-key", null, {}, io(), { chatModel: fake.chatModel }, stop.signal);
    while (fake.requests.length === 0) await new Promise((resolve) => setTimeout(resolve, 1));
    stop.abort(reason);
    await expect(checking).rejects.toBe(reason);
  });
});

/**
 * `setupAuth` with a fake `listModels` and a fake `chatModel`. The listing
 * records `[provider, apiKey, baseUrl]`, the headers and the signal of each call, then waits for
 * its signal when `wait` is set, throws `models` when it is an error, or answers it. Each reply
 * check's config is in `chat.built`; `answers()` gives `[provider, model, api_key, base_url]` per
 * check and `answerHeaders()` the headers.
 */
function checks() {
  const setup = setupAuth();
  const chat = fakeChatModel();
  const fake = {
    models: ["m1", "m2"] as readonly string[] | Error,
    wait: false,
    listings: [] as [unknown, unknown, unknown][],
    listingHeaders: [] as Record<string, string>[],
    listingSignals: [] as (AbortSignal | undefined)[],
  };
  const listModels = (async (provider: unknown, options: ListModelsOptions = {}) => {
    fake.listings.push([provider, options.apiKey ?? null, options.baseUrl ?? null]);
    fake.listingHeaders.push({ ...options.headers });
    fake.listingSignals.push(options.signal);
    if (fake.wait) {
      const signal = options.signal!;
      await new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    }
    if (fake.models instanceof Error) throw fake.models;
    return fake.models;
  }) as NonNullable<AuthDeps["listModels"]>;
  const answers = () =>
    chat.built.map(({ config }) => [config["provider"], config["model"], config["api_key"] ?? null, config["base_url"] ?? null]);
  const answerHeaders = () => chat.built.map(({ config }) => config["headers"] ?? {});
  const deps: AuthDeps = { ...setup.deps, listModels, chatModel: chat.chatModel };
  return { ...setup, deps, fake, chat, answers, answerHeaders };
}

describe("auth provider setup", () => {
  /** Run `yian auth provider options` on `c`'s store and environment, `deps` overriding. */
  async function runAuth(
    c: ReturnType<typeof checks>,
    provider: string,
    options: Record<string, OptionValue> = {},
    deps: AuthDeps = {},
  ) {
    const io = fakeIo({ env: c.env });
    const result = await run(auth, authArgs(provider, options), io, { ...c.deps, ...deps });
    return { ...result, output: result.out + result.err };
  }

  const MODEL = { kind: "text", label: "Model:", secret: false };

  // Api key provider setup saves key and model.
  test.each(
    Object.keys(PROVIDER_API_KEY_ENV)
      .filter((name) => !["openai", "azure", "ollama", "vllm"].includes(name))
      .sort(),
  )("%s setup saves the key and model", async (provider) => {
    const c = checks();
    c.env[PROVIDER_API_KEY_ENV[provider]!] = "private-key";
    const result = await runAuth(c, provider, { model: "m1" });
    expect(result).toMatchObject({
      code: 0,
      out: `Saved ${provider} login (api-key) with model m1\nDefault provider: ${provider}\n`,
      err: "",
    });
    const selected = c.store.selection(provider)!;
    expect([selected.method, selected.model]).toEqual(["api_key", "m1"]);
    expect(((await c.store.credentials(provider, selected.credentialId)) as ApiKeyCredentials).apiKey).toBe(
      "private-key",
    );
    expect(c.store.read().defaultProvider).toBe(provider);
    expect(c.fake.listings).toEqual([[provider, "private-key", null]]);
    expect(c.fake.listingSignals[0]).toBeInstanceOf(AbortSignal);
    expect(c.answers()).toEqual([]);
  });

  // Api key setup keeps another default.
  test("API-key setup keeps another default", async () => {
    const c = checks();
    await c.store.save("openai", new ApiKeyCredentials("private-other"), "gpt-test");
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    const result = await runAuth(c, "anthropic", { model: "m2" });
    expect(result.code).toBe(0);
    expect(c.store.selection("anthropic")?.model).toBe("m2");
    expect(c.store.read().defaultProvider).toBe("openai");
    expect(result.out).toContain("Default provider: openai (switch with 'yian auth anthropic --default')\n");
  });

  // Api key env option names the variable.
  test("--api-key-env names the key variable", async () => {
    const c = checks();
    let result = await runAuth(c, "anthropic", { model: "m1" });
    expect(result).toMatchObject({
      code: 1,
      err: "Set ANTHROPIC_API_KEY before API-key setup, or run in an interactive terminal\n",
    });
    expect(c.fake.listings).toEqual([]);
    expect(existsSync(c.path)).toBe(false);

    c.env["MY_KEY"] = "private-key";
    result = await runAuth(c, "anthropic", { model: "m1", "api-key-env": "MY_KEY" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([["anthropic", "private-key", null]]);
  });

  // Unlisted model is checked with one reply.
  test("an unlisted model is checked with one reply", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.models = ["claude-x-20250101"];
    let result = await runAuth(c, "anthropic", { model: "claude-x" });
    expect(result.code).toBe(0);
    expect(c.answers()).toEqual([["anthropic", "claude-x", "private-key", null]]);
    expect(c.store.selection("anthropic")?.model).toBe("claude-x");

    const before = readFileSync(c.path);
    c.chat.failure = named("NotFoundError", "model not found");
    result = await runAuth(c, "anthropic", { model: "claude-y" });
    expect(result).toMatchObject({
      code: 1,
      err: "The model didn't answer: NotFoundError: model not found; the previous login is unchanged\n",
    });
    expect(readFileSync(c.path)).toEqual(before);
  });

  const raw = "m\x1b]52;c;aGk=\x07-a\x1b[2J";
  test.each<[string, boolean, string]>([
    ["stripped on a TTY", true, "m-a"],
    ["kept exactly when piped", false, raw],
  ])("a provider's error is %s", async (_, outTty, shown) => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.models = [];
    c.chat.failure = named("NotFoundError", raw);
    const io = fakeIo({ env: c.env, outTty });
    const result = await run(auth, authArgs("anthropic", { model: "claude-y" }), io, c.deps);
    expect(result.err).toBe(`The model didn't answer: NotFoundError: ${shown}; the previous login is unchanged\n`);
  });

  // Listing failure falls back to one reply.
  test("a listing failure falls back to one reply", async () => {
    const c = checks();
    const previous = await c.store.save("anthropic", new ApiKeyCredentials("old-key"), "claude-old");
    const before = readFileSync(c.path);
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.models = new ModelListError("HTTP 401");
    c.chat.failure = named("AuthenticationError", "invalid key");
    const result = await runAuth(c, "anthropic", { model: "claude-x" });
    expect(result.code).toBe(1);
    expect(c.answers()).toEqual([["anthropic", "claude-x", "private-key", null]]);
    expect(readFileSync(c.path)).toEqual(before);
    expect(c.store.selection()).toEqual(previous);
    expect(c.store.read().defaultProvider).toBe("anthropic");
    expect(result.output).not.toContain("private-");
    expect(result.output).not.toContain("old-key");
  });

  // Listing failure on a terminal asks for a model name.
  test("a listing failure on a terminal asks for a model name", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.models = new ModelListError("can't reach host");
    const fake = fakeTerminal(["claude-x"]);
    const result = await runAuth(c, "anthropic", {}, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(result.out).toStartWith("Couldn't list the anthropic models (can't reach host); type a model name\n");
    expect(fake.asked).toEqual([MODEL]);
    expect(c.answers()).toEqual([["anthropic", "claude-x", "private-key", null]]);
    expect(c.store.selection("anthropic")?.model).toBe("claude-x");
  });

  test("an empty listing on a terminal asks for a model name", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.models = [];
    const fake = fakeTerminal(["claude-x"]);
    const result = await runAuth(c, "anthropic", {}, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(result.out).toStartWith("Couldn't list the anthropic models (none were returned); type a model name\n");
  });

  test("a listed model is chosen on a terminal, starting at the first", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    const fake = fakeTerminal(["m2"]);
    const result = await runAuth(c, "anthropic", {}, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([{ kind: "choice", label: "Model:", choices: ["m1", "m2"], initial: "m1" }]);
    expect(c.answers()).toEqual([]);
    expect(c.store.selection("anthropic")?.model).toBe("m2");
  });

  test.each<[string, string | null]>([
    ["an empty", "  "],
    ["a cancelled", null],
  ])("%s model name saves nothing", async (_, answer) => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.models = [];
    const result = await runAuth(c, "anthropic", {}, { terminal: fakeTerminal([answer]).terminal });
    expect(result).toMatchObject({ code: 1, err: `${CANCELLED}\n` });
    expect(c.answers()).toEqual([]);
    expect(existsSync(c.path)).toBe(false);
  });

  // Base url variable skips the listing.
  test.each([
    ["anthropic", "ANTHROPIC_BASE_URL"],
    ["anthropic", "ANTHROPIC_API_URL"],
    ["google", "GOOGLE_GEMINI_BASE_URL"],
  ])("%s with %s set skips the listing", async (provider, variable) => {
    const c = checks();
    c.env[PROVIDER_API_KEY_ENV[provider]!] = "private-key";
    c.env[variable] = "https://gateway.example.com";
    const result = await runAuth(c, provider, { model: "m1" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([]);
    expect(c.answers()).toEqual([[provider, "m1", "private-key", null]]);
    expect(c.store.selection(provider)?.model).toBe("m1");
  });

  // Vertex switch does not skip the google listing.
  test("the Vertex switch doesn't skip the Google listing", async () => {
    const c = checks();
    c.env["GOOGLE_API_KEY"] = "private-key";
    c.env["GOOGLE_GENAI_USE_VERTEXAI"] = "true";
    const result = await runAuth(c, "google", { model: "m1" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([["google", "private-key", null]]);
    expect(c.answers()).toEqual([]);
    expect(c.store.selection("google")?.model).toBe("m1");
  });

  test("Google setup reads GEMINI_API_KEY when GOOGLE_API_KEY is unset", async () => {
    const c = checks();
    let result = await runAuth(c, "google", { model: "m1" });
    expect(result).toMatchObject({
      code: 1,
      err: "Set GOOGLE_API_KEY or GEMINI_API_KEY before API-key setup, or run in an interactive terminal\n",
    });
    c.env["GEMINI_API_KEY"] = "private-gemini";
    result = await runAuth(c, "google", { model: "m1" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([["google", "private-gemini", null]]);
  });

  // Base url variable on a terminal asks for a model name.
  test("a base-URL variable on a terminal asks for a model name", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.env["ANTHROPIC_BASE_URL"] = "https://gateway.example.com";
    const fake = fakeTerminal(["claude-x"]);
    const result = await runAuth(c, "anthropic", {}, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(result.out).toStartWith(
      "ANTHROPIC_BASE_URL is set, so runs use another host than the anthropic listing; type a model name\n",
    );
    expect(fake.asked).toEqual([MODEL]);
    expect(c.fake.listings).toEqual([]);
    expect(c.answers()).toEqual([["anthropic", "claude-x", "private-key", null]]);
  });

  // Ollama setup saves model only.
  test("ollama setup saves the model only", async () => {
    const c = checks();
    c.env["OLLAMA_BASE_URL"] = "http://localhost:9999";
    c.fake.models = ["llama3"];
    const result = await runAuth(c, "ollama", { model: "llama3" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([["ollama", null, "http://localhost:9999"]]);
    expect(c.answers()).toEqual([]);
    const selected = c.store.selection("ollama")!;
    expect([selected.method, selected.model]).toEqual(["none", "llama3"]);
    expect(result.out).toContain("Saved ollama login (none) with model llama3\n");
  });

  // Vllm key is optional.
  test("vllm's key is optional", async () => {
    const c = checks();
    let result = await runAuth(c, "vllm", { model: "m1" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([["vllm", null, null]]);
    expect(c.store.selection("vllm")?.method).toBe("none");

    c.env["VLLM_API_KEY"] = "private-key";
    c.env["VLLM_BASE_URL"] = "https://vllm.example.com/v1";
    result = await runAuth(c, "vllm", { model: "m1" });
    expect(result.code).toBe(0);
    expect(c.fake.listings.at(-1)).toEqual(["vllm", "private-key", "https://vllm.example.com/v1"]);
    expect(c.store.selection("vllm")?.method).toBe("api_key");
    expect(result.output).not.toContain("private-");
  });

  // Vllm named key variable is required.
  test("a vllm key variable named by --api-key-env is required", async () => {
    const c = checks();
    const result = await runAuth(c, "vllm", { model: "m1", "api-key-env": "MY_KEY" });
    expect(result).toMatchObject({ code: 1, err: "Set MY_KEY before API-key setup, or run in an interactive terminal\n" });
    expect(c.fake.listings).toEqual([]);
    expect(existsSync(c.path)).toBe(false);
  });

  // Hidden prompt reads the key.
  test("a hidden prompt reads the key", async () => {
    const c = checks();
    let fake = fakeTerminal(["private-key"]);
    let result = await runAuth(c, "anthropic", { model: "m1" }, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([{ kind: "text", label: "anthropic API key:", secret: true }]);
    expect(c.fake.listings).toEqual([["anthropic", "private-key", null]]);
    expect(c.store.selection("anthropic")?.method).toBe("api_key");

    fake = fakeTerminal([null]);
    result = await runAuth(c, "anthropic", { model: "m2" }, { terminal: fake.terminal });
    expect(result).toMatchObject({ code: 1, err: `${CANCELLED}\n` });
    expect(c.store.selection("anthropic")?.model).toBe("m1");
  });

  // Hidden prompt empty answer means no vllm key.
  test("an empty hidden-prompt answer means no vllm key", async () => {
    const c = checks();
    let fake = fakeTerminal([""]);
    let result = await runAuth(c, "vllm", { model: "m1" }, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([{ kind: "text", label: "vllm API key (leave empty for none):", secret: true }]);
    expect(c.store.selection("vllm")?.method).toBe("none");

    fake = fakeTerminal([""]);
    result = await runAuth(c, "vllm", { model: "m2", "api-key-env": "MY_KEY" }, { terminal: fake.terminal });
    expect(result).toMatchObject({ code: 1, err: "Authentication cancelled; no API key was supplied\n" });
    expect(fake.asked).toEqual([{ kind: "text", label: "vllm API key:", secret: true }]);
    expect(c.store.selection("vllm")?.model).toBe("m1");
  });

  // Noninteractive setup requires model.
  test("non-interactive setup requires --model", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    for (const provider of ["anthropic", "ollama"]) {
      const result = await runAuth(c, provider);
      expect(result).toMatchObject({
        code: 1,
        err: "Authentication needs --model when no interactive terminal is available\n",
      });
    }
    expect(c.fake.listings).toEqual([]);
    expect(existsSync(c.path)).toBe(false);
  });

  // Azure setup saves endpoint key and deployment.
  test("azure setup saves the endpoint, key and deployment", async () => {
    const c = checks();
    c.env["AZURE_OPENAI_API_KEY"] = "private-key";
    c.env["OPENAI_API_VERSION"] = "2025-03-01-preview";
    const endpoint = "https://r.openai.azure.com/";
    let result = await runAuth(c, "azure", { endpoint, model: "dep" });
    expect(result.code).toBe(0);
    expect(c.fake.listings).toEqual([]);
    expect(c.answers()).toEqual([["azure", "dep", "private-key", endpoint]]);
    const selected = c.store.selection("azure")!;
    expect([selected.endpoint, selected.model, selected.method]).toEqual([endpoint, "dep", "api_key"]);
    expect(result.output).not.toContain("private-");

    const before = readFileSync(c.path);
    result = await runAuth(c, "azure", { model: "dep" });
    expect(result).toMatchObject({ code: 1, err: "Authentication needs --endpoint when no interactive terminal is available\n" });
    result = await runAuth(c, "azure", { endpoint });
    expect(result).toMatchObject({ code: 1, err: "Authentication needs --model when no interactive terminal is available\n" });
    result = await runAuth(c, "azure", { endpoint: "http://r.example.com", model: "dep" });
    expect(result.code).toBe(1);
    expect(result.err).toStartWith("Invalid Azure endpoint: ");
    expect(c.answers()).toHaveLength(1);
    expect(c.fake.listings).toEqual([]);
    expect(readFileSync(c.path)).toEqual(before);
  });

  test("azure setup asks for the endpoint and deployment on a terminal", async () => {
    const c = checks();
    c.env["AZURE_OPENAI_API_KEY"] = "private-key";
    c.env["OPENAI_API_VERSION"] = "2025-03-01-preview";
    const endpoint = "https://r.openai.azure.com/";
    const fake = fakeTerminal([` ${endpoint} `, "dep"]);
    const result = await runAuth(c, "azure", {}, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([
      { kind: "text", label: "Azure OpenAI endpoint (https://<resource>.openai.azure.com/):", secret: false },
      { kind: "text", label: "Deployment name:", secret: false },
    ]);
    expect(c.answers()).toEqual([["azure", "dep", "private-key", endpoint]]);
    expect(c.store.selection("azure")?.endpoint).toBe(endpoint);
  });

  // Azure setup requires the api version.
  test("azure setup requires the API version", async () => {
    const c = checks();
    const result = await runAuth(c, "azure", { endpoint: "https://r.openai.azure.com/", model: "dep" });
    expect(result).toMatchObject({ code: 1, err: "Azure needs OPENAI_API_VERSION set to its API version (runs read it too)\n" });
    expect(result.output).not.toContain("AZURE_OPENAI_API_KEY");
    expect(c.answers()).toEqual([]);
    expect(c.fake.listings).toEqual([]);
    expect(existsSync(c.path)).toBe(false);
  });

  // Provider setup sends headers to listing and reply check.
  test("provider setup sends the headers to the listing and the reply check", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.env["GW_TOKEN"] = "private-gw";
    const result = await runAuth(c, "anthropic", { header: ["X-G=GW_TOKEN"], model: "m" });
    expect(result.code).toBe(0);
    expect(c.fake.listingHeaders).toEqual([{ "X-G": "private-gw" }]);
    expect(c.answerHeaders()).toEqual([{ "X-G": "private-gw" }]);
    const selected = c.store.selection("anthropic")!;
    expect(selected.headerNames).toEqual(["X-G"]);
    expect(selected.organization).toBeNull();
    const credentials = (await c.store.credentials("anthropic", selected.credentialId)) as ApiKeyCredentials;
    expect(credentials.headers).toEqual([["X-G", "private-gw"]]);
    expect(result.output).not.toContain("private-");
  });

  // Saving again without settings keeps none.
  test("saving again without settings keeps none", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.env["GW_TOKEN"] = "private-gw";
    expect((await runAuth(c, "anthropic", { header: ["X-G=GW_TOKEN"], model: "m1" })).code).toBe(0);
    expect((await runAuth(c, "anthropic", { model: "m1" })).code).toBe(0);
    const selected = c.store.selection("anthropic")!;
    expect(selected.headerNames).toEqual([]);
    expect(((await c.store.credentials("anthropic", selected.credentialId)) as ApiKeyCredentials).headers).toEqual([]);
    expect(c.fake.listingHeaders.at(-1)).toEqual({});
  });

  // Bare header reads a hidden prompt.
  test("a bare header reads a hidden prompt", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    let fake = fakeTerminal(["private-gw"]);
    let result = await runAuth(c, "anthropic", { header: ["X-G"], model: "m1" }, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(fake.asked).toEqual([{ kind: "text", label: "X-G header value:", secret: true }]);
    expect(c.fake.listingHeaders).toEqual([{ "X-G": "private-gw" }]);
    expect(c.store.selection("anthropic")?.headerNames).toEqual(["X-G"]);
    expect(result.output).not.toContain("private-");

    fake = fakeTerminal(["private bad "]);
    result = await runAuth(c, "anthropic", { header: ["X-G"], model: "m2" }, { terminal: fake.terminal });
    expect(result.code).toBe(1);
    expect(result.err).toStartWith("Invalid --header: ");
    expect(result.err).toContain("X-G");
    expect(result.output).not.toContain("private");
    expect(c.store.selection("anthropic")?.model).toBe("m1");
  });

  // Headers need a vllm key.
  test("headers need a vllm key", async () => {
    const c = checks();
    c.env["GW_TOKEN"] = "private-gw";
    let result = await runAuth(c, "vllm", { header: ["X-G=GW_TOKEN"], model: "m1" });
    expect(result).toMatchObject({
      code: 1,
      err: "Set VLLM_API_KEY before API-key setup, or run in an interactive terminal\n",
    });
    expect(c.fake.listings).toEqual([]);
    expect(existsSync(c.path)).toBe(false);

    c.env["VLLM_API_KEY"] = "private-key";
    c.env["VLLM_BASE_URL"] = "https://vllm.example.com/v1";
    result = await runAuth(c, "vllm", { header: ["X-G=GW_TOKEN"], model: "m1" });
    expect(result.code).toBe(0);
    expect(c.store.selection("vllm")?.headerNames).toEqual(["X-G"]);
  });

  // Listing failure text is scrubbed of header values.
  test("the listing failure text is scrubbed of header values", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.env["GW_TOKEN"] = "gwvalue7";
    c.fake.models = new ModelListError("rejected gwvalue7");
    const fake = fakeTerminal(["claude-x"]);
    const result = await runAuth(c, "anthropic", { header: ["X-G=GW_TOKEN"] }, { terminal: fake.terminal });
    expect(result.code).toBe(0);
    expect(result.out).toContain("(rejected [REDACTED])");
    expect(result.output).not.toContain("gwvalue7");
  });

  test("a SIGINT during the reply check cancels the setup and saves nothing", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.chat.hang = true;
    const signals = new EventEmitter();
    const io = fakeIo({ env: c.env });
    const running = auth(authArgs("anthropic", { model: "claude-x" }), io.io, { ...c.deps, signals });
    while (c.chat.requests.length === 0) await new Promise((resolve) => setTimeout(resolve, 1));
    signals.emit("SIGINT");
    expect({ code: await running, out: io.out(), err: io.err() }).toEqual({ code: 1, out: "", err: `${CANCELLED}\n` });
    expect(signals.listenerCount("SIGINT")).toBe(0);
    expect(existsSync(c.path)).toBe(false);
  });

  test("a SIGINT during the listing cancels the setup before any prompt and saves nothing", async () => {
    const c = checks();
    c.env["ANTHROPIC_API_KEY"] = "private-key";
    c.fake.wait = true;
    const signals = new EventEmitter();
    const fake = fakeTerminal([]);
    const io = fakeIo({ env: c.env });
    const running = auth(authArgs("anthropic"), io.io, { ...c.deps, terminal: fake.terminal, signals });
    while (c.fake.listings.length === 0) await new Promise((resolve) => setTimeout(resolve, 1));
    signals.emit("SIGINT");
    expect({ code: await running, out: io.out(), err: io.err() }).toEqual({ code: 1, out: "", err: `${CANCELLED}\n` });
    expect(c.fake.listingSignals[0]?.aborted).toBe(true);
    expect(fake.asked).toEqual([]);
    expect(c.answers()).toEqual([]);
    expect(existsSync(c.path)).toBe(false);
  });
});

describe("openBrowser", () => {
  const URL = "https://auth.example.invalid/authorize?a=1&b=2";
  const SHOWN = `Open this URL to sign in:\n${URL}\n`;

  /**
   * A fake `launch` recording each call and what `out` held then; its child records `unref()` and
   * emits `event` (with an error for "error") just after the call returns, or nothing for `null`.
   */
  function fakeLaunch(out: ReturnType<typeof sink>, event: "spawn" | "error" | null) {
    const launched: { command: string; args: readonly string[]; options: object; printed: string }[] = [];
    const child = Object.assign(new EventEmitter(), { unrefs: 0, unref: () => void (child.unrefs += 1) });
    const launch = (command: string, args: readonly string[], options: object) => {
      launched.push({ command, args, options, printed: out.text() });
      if (event !== null) {
        queueMicrotask(() => child.emit(event, ...(event === "error" ? [new Error("spawn ENOENT")] : [])));
      }
      return child;
    };
    return { launch, launched, child };
  }

  test.each<[NodeJS.Platform, string, string[]]>([
    ["darwin", "open", [URL]],
    ["linux", "xdg-open", [URL]],
    ["freebsd", "xdg-open", [URL]],
    ["win32", "rundll32", ["url.dll,FileProtocolHandler", URL]],
  ])("on %s it prints the URL, then runs %s without a shell, detached, unref'd", async (platform, command, args) => {
    const out = sink();
    const fake = fakeLaunch(out, "spawn");
    await openBrowser(URL, out.out, { platform, launch: fake.launch });
    expect(fake.launched).toEqual([{ command, args, options: { detached: true, stdio: "ignore" }, printed: SHOWN }]);
    expect(fake.child.unrefs).toBe(1);
    expect(out.text()).toBe(SHOWN);
  });

  test("resolves only once the opener has spawned, and a later error doesn't throw", async () => {
    const out = sink();
    const fake = fakeLaunch(out, null);
    let resolved = false;
    const opening = openBrowser(URL, out.out, { platform: "linux", launch: fake.launch }).then(() => {
      resolved = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fake.launched).toHaveLength(1);
    expect(resolved).toBe(false);
    fake.child.emit("spawn");
    await opening;
    expect(resolved).toBe(true);
    // The error listener stays attached, so a late error doesn't end the process.
    expect(() => fake.child.emit("error", new Error("late"))).not.toThrow();
  });

  test("an opener that fails to start still resolves", async () => {
    const out = sink();
    const fake = fakeLaunch(out, "error");
    await openBrowser(URL, out.out, { platform: "linux", launch: fake.launch });
    expect(fake.launched).toHaveLength(1);
    expect(out.text()).toBe(SHOWN);
  });

  test("a launch that throws still resolves after printing the URL", async () => {
    const out = sink();
    await openBrowser(URL, out.out, {
      platform: "linux",
      launch: () => {
        throw new Error("no opener");
      },
    });
    expect(out.text()).toBe(SHOWN);
  });
});
