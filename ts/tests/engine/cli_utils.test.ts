// Ported from tests/engine/test_cli_utils.py: the key and endpoint helpers, with a fake terminal
// standing in for questionary and a TTY; plus the argument parser, the usage text and `printJson`.
// The `_agent_connections` / `_ensure_provider_keys` tests come with `yian run`.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Writable } from "node:stream";

import {
  CommandEnv,
  type CommandSpec,
  confirmOllamaEndpoint,
  ensureApiKey,
  ensureEnvKey,
  flush,
  helpText,
  JSON_FORMAT,
  type OptionSpec,
  type ParsedArgs,
  parseCommand,
  printJson,
  streamWriter,
  UsageError,
  usageError,
  type Writer,
} from "../../src/cli/utils.ts";
import type { Env } from "../../src/defaults.ts";
import { reprText } from "../../src/diagnostics.ts";
import { ApiKeyCredentials, AuthStore, OAuthCredentials } from "../../src/llm_clients/auth/index.ts";

// Each test's environment: a private home and auth store, so no real login is read.
let home: string;
let vars: Map<string, string>;
const base: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-cli-utils-"));
  vars = new Map([
    ["HOME", home],
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
  ]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

interface Prompt {
  label: string;
  initial?: string;
  secret?: boolean;
}

/** A fake terminal answering `askText` from `answers` (one per prompt) or with one answer always. */
function fakeTerminal(answers: (string | null)[] | string | null) {
  const prompts: Prompt[] = [];
  const queue = Array.isArray(answers) ? [...answers] : null;
  return {
    prompts,
    askText: async (label: string, options: { initial?: string; secret?: boolean } = {}) => {
      prompts.push({ label, ...options });
      if (queue === null) return answers as string | null;
      if (queue.length === 0) throw new Error("no answer left");
      return queue.shift()!;
    },
  };
}

/** Return the error `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

/** Return the error `run` throws. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

describe("ensureApiKey", () => {
  test("returns an existing key", async () => {
    vars.set("ANTHROPIC_API_KEY", "sk-existing");
    expect(await ensureApiKey("anthropic", { env: new CommandEnv(base), terminal: null })).toBe("sk-existing");
  });

  test("a keyless provider gives null", async () => {
    expect(await ensureApiKey("ollama", { env: new CommandEnv(base), terminal: null })).toBeNull();
  });

  test("an unknown provider gives null", async () => {
    expect(await ensureApiKey("totally-made-up", { env: new CommandEnv(base), terminal: null })).toBeNull();
  });

  test("matches the provider case-insensitively", async () => {
    vars.set("OPENAI_API_KEY", "sk-oai");
    expect(await ensureApiKey("OpenAI", { env: new CommandEnv(base), terminal: null })).toBe("sk-oai");
  });

  test("a missing key without a terminal throws", async () => {
    const error = await rejection(ensureApiKey("anthropic", { env: new CommandEnv(base), terminal: null }));
    expect(error.message).toContain("ANTHROPIC_API_KEY");
  });

  test("prompts and sets the typed key", async () => {
    const env = new CommandEnv(base);
    const terminal = fakeTerminal("sk-typed");
    expect(await ensureApiKey("anthropic", { env, terminal })).toBe("sk-typed");
    expect(env.read("ANTHROPIC_API_KEY")).toBe("sk-typed");
    expect(terminal.prompts.some((p) => p.label.includes("ANTHROPIC_API_KEY"))).toBe(true);
  });

  test("an empty answer re-prompts", async () => {
    const env = new CommandEnv(base);
    const terminal = fakeTerminal(["", "sk-typed"]);
    expect(await ensureApiKey("anthropic", { env, terminal })).toBe("sk-typed");
    expect(env.read("ANTHROPIC_API_KEY")).toBe("sk-typed");
    expect(terminal.prompts).toHaveLength(2);
  });

  test("a cancel throws at once", async () => {
    const terminal = fakeTerminal([null]);
    const error = await rejection(ensureApiKey("anthropic", { env: new CommandEnv(base), terminal }));
    expect(error.message).toContain("no API key provided");
    expect(terminal.prompts).toHaveLength(1);
  });

  test("three empty answers exhaust the prompts", async () => {
    const terminal = fakeTerminal(["", "", ""]);
    const error = await rejection(ensureApiKey("anthropic", { env: new CommandEnv(base), terminal }));
    expect(error.message).toContain("after 3 attempts");
    expect(terminal.prompts).toHaveLength(3);
  });

  test("the last attempt can succeed", async () => {
    const terminal = fakeTerminal(["  ", "", "sk-typed"]);
    expect(await ensureApiKey("anthropic", { env: new CommandEnv(base), terminal })).toBe("sk-typed");
    expect(terminal.prompts).toHaveLength(3);
  });

  test("strips the surrounding whitespace", async () => {
    const env = new CommandEnv(base);
    expect(await ensureApiKey("anthropic", { env, terminal: fakeTerminal(["  sk-abc  "]) })).toBe("sk-abc");
    expect(env.read("ANTHROPIC_API_KEY")).toBe("sk-abc");
  });

  test("without a terminal it suggests a saved login", async () => {
    const error = await rejection(ensureApiKey("openai", { env: new CommandEnv(base), terminal: null }));
    expect(error.message).toMatch(/OPENAI_API_KEY.*run 'yian auth openai'/);
  });

  test("a saved login needs no key prompt", async () => {
    await new AuthStore(null, { env: base }).save("anthropic", new ApiKeyCredentials("saved-k"), "m", {
      makeDefault: false,
    });
    const terminal = fakeTerminal("typed");
    expect(await ensureApiKey("anthropic", { env: new CommandEnv(base), terminal })).toBe("saved-k");
    expect(terminal.prompts).toEqual([]);
  });

  test("a saved subscription gives null without a prompt", async () => {
    await new AuthStore(null, { env: base }).save(
      "openai",
      new OAuthCredentials({
        accessToken: "access",
        refreshToken: "refresh",
        expiresAt: Date.now() / 1000 + 3600,
        accountId: "acct",
      }),
      "gpt-x",
    );
    const terminal = fakeTerminal("typed");
    expect(await ensureApiKey("openai", { env: new CommandEnv(base), terminal })).toBeNull();
    expect(terminal.prompts).toEqual([]);
  });
});

describe("ensureEnvKey", () => {
  test("returns an existing value", async () => {
    vars.set("OPENAI_API_KEY", "sk-oai");
    expect(await ensureEnvKey("openai", { env: new CommandEnv(base), terminal: null })).toBe("sk-oai");
  });

  test("keyless and unknown providers give null", async () => {
    const env = new CommandEnv(base);
    expect(await ensureEnvKey("ollama", { env, terminal: null })).toBeNull();
    expect(await ensureEnvKey("totally-made-up", { env, terminal: null })).toBeNull();
  });

  test("matches the provider case-insensitively", async () => {
    vars.set("OPENAI_API_KEY", "sk-oai");
    expect(await ensureEnvKey("OPENAI", { env: new CommandEnv(base), terminal: null })).toBe("sk-oai");
  });

  test("a saved login is not consulted", async () => {
    await new AuthStore(null, { env: base }).save("anthropic", new ApiKeyCredentials("saved-k"), "m");
    const error = await rejection(ensureEnvKey("anthropic", { env: new CommandEnv(base), terminal: null }));
    expect(error.message).toMatch(/ANTHROPIC_API_KEY.*run 'yian auth anthropic'/);
  });

  test("without a terminal the message names the variable and the login command", async () => {
    const error = await rejection(ensureEnvKey("OpenAI", { env: new CommandEnv(base), terminal: null }));
    expect(error.message).toBe(
      "provider 'OpenAI' needs an API key but $OPENAI_API_KEY is not set (set it in the environment, " +
        "or run interactively to be prompted). Or run 'yian auth openai' to save a login",
    );
  });

  test("prompts with a secret field, strips, sets and returns the key", async () => {
    const env = new CommandEnv(base);
    const terminal = fakeTerminal([" sk-new "]);
    expect(await ensureEnvKey("openai", { env, terminal })).toBe("sk-new");
    expect(terminal.prompts).toEqual([{ label: "Enter OPENAI_API_KEY for provider 'openai':", secret: true }]);
    expect(env.read("OPENAI_API_KEY")).toBe("sk-new");
    expect(vars.has("OPENAI_API_KEY")).toBe(false);
  });

  test("empty answers end after the attempts", async () => {
    const terminal = fakeTerminal(["", " ", ""]);
    const error = await rejection(ensureEnvKey("openai", { env: new CommandEnv(base), terminal }));
    expect(error.message).toBe("no API key provided for 'openai' ($OPENAI_API_KEY) after 3 attempts");
  });

  test("a cancel throws at once", async () => {
    const terminal = fakeTerminal([null]);
    const error = await rejection(ensureEnvKey("openai", { env: new CommandEnv(base), terminal }));
    expect(error.message).toBe("no API key provided for 'openai' ($OPENAI_API_KEY)");
    expect(terminal.prompts).toHaveLength(1);
  });

  test("skips an unset optional key", async () => {
    const terminal = fakeTerminal("typed");
    expect(await ensureEnvKey("vllm", { env: new CommandEnv(base), terminal })).toBeNull();
    expect(terminal.prompts).toEqual([]);
    vars.set("VLLM_API_KEY", "sk-vllm");
    expect(await ensureEnvKey("vllm", { env: new CommandEnv(base), terminal: null })).toBe("sk-vllm");
  });
});

describe("confirmOllamaEndpoint", () => {
  test("without a terminal returns the default", async () => {
    expect(await confirmOllamaEndpoint({ env: new CommandEnv(base), terminal: null })).toBe("http://localhost:11434");
  });

  test("without a terminal returns the environment's value", async () => {
    vars.set("OLLAMA_BASE_URL", "http://gpu-box:11434");
    expect(await confirmOllamaEndpoint({ env: new CommandEnv(base), terminal: null })).toBe("http://gpu-box:11434");
  });

  test("prompts and sets the answer", async () => {
    const env = new CommandEnv(base);
    const terminal = fakeTerminal(" http://h200:11434 ");
    expect(await confirmOllamaEndpoint({ env, terminal })).toBe("http://h200:11434");
    expect(env.read("OLLAMA_BASE_URL")).toBe("http://h200:11434");
    expect(terminal.prompts).toEqual([{ label: "Ollama endpoint:", initial: "http://localhost:11434" }]);
  });

  test("an empty answer keeps the current value", async () => {
    vars.set("OLLAMA_BASE_URL", "http://gpu-box:11434");
    const env = new CommandEnv(base);
    expect(await confirmOllamaEndpoint({ env, terminal: fakeTerminal("") })).toBe("http://gpu-box:11434");
    expect(env.read("OLLAMA_BASE_URL")).toBe("http://gpu-box:11434");
  });
});

describe("CommandEnv", () => {
  test("a set name shadows the base; others fall through", () => {
    vars.set("A", "base-a");
    vars.set("B", "base-b");
    const env = new CommandEnv(base);
    env.set("A", "set-a");
    const read = env.read;
    expect(read("A")).toBe("set-a");
    expect(read("B")).toBe("base-b");
    expect(read("C")).toBeUndefined();
  });
});

const RUN: CommandSpec = {
  name: "run",
  summary: "Start or recover a durable flow run and print its terminal output.",
  positionals: [{ name: "flow", help: "Path to a flow .yaml", path: { exists: true, dir: false, readable: true } }],
  options: [
    { name: "input", short: "i", type: "string", multiple: true, help: "An input as k=v (repeatable)." },
    {
      name: "resume",
      type: "string",
      path: { exists: true, dir: false, readable: true, writable: true },
      help: "Recover the run recorded at PATH.",
    },
    { name: "log", type: "string", path: { dir: false }, help: "Create a fresh durable run log at PATH." },
    { name: "memory-root", type: "string", path: { dir: true }, metavar: "DIR", help: "Keep memory under DIR." },
    { name: "quiet", short: "q", type: "boolean", help: "Suppress per-node progress." },
    { name: "num-workers", short: "w", type: "int", min: 0, help: "Worker pool size." },
    { name: "json", type: "boolean", help: "Print one yian.cli.v1 JSON document on stdout." },
  ],
};

const CHECK: CommandSpec = {
  name: "check",
  summary: "Load each flow without running it and report whether it loads.",
  positionals: [{ name: "flows", help: "Paths to flow .yaml files.", variadic: true }],
  options: [
    { name: "json", type: "boolean", help: "Print one yian.cli.v1 JSON document on stdout." },
    { name: "root", type: "string", path: { exists: true, dir: true }, help: "Load flows only from here." },
  ],
};

describe("parseCommand", () => {
  let flow: string;
  beforeEach(() => {
    flow = join(home, "flow.yaml");
    writeFileSync(flow, "flow: {}\n");
  });

  /** Return the usage error message `argv` gives for `spec`. */
  function usage(spec: CommandSpec, argv: string[]): string {
    const error = thrown(() => parseCommand(spec, argv));
    expect(error).toBeInstanceOf(UsageError);
    return error.message;
  }

  test("long and short options, --opt=value, repeatable options, ints and booleans", () => {
    expect(parseCommand(RUN, ["-i", "a=1", "--input=b=2", "--input", "c=3", "-w", "0", "-q", "--json", flow])).toEqual({
      help: false,
      args: { flow },
      options: { input: ["a=1", "b=2", "c=3"], "num-workers": 0, quiet: true, json: true },
    });
    expect(parseCommand(RUN, [flow, "--num-workers=12"])).toEqual({
      help: false,
      args: { flow },
      options: { input: [], "num-workers": 12, quiet: false, json: false },
    });
  });

  test("int options are checked", () => {
    expect(usage(RUN, [flow, "-w", "x"])).toBe("Invalid value for '--num-workers' / '-w': 'x' is not a valid integer.");
    for (const argv of [["--num-workers=-1"], ["-w-1"]]) {
      expect(usage(RUN, [flow, ...argv])).toBe("Invalid value for '--num-workers' / '-w': -1 is not in the range x>=0.");
    }
    // parseArgs refuses a value starting with `-` written as a separate word.
    expect(usage(RUN, [flow, "-w", "-1"])).toBe("Option '-w' requires an argument.");
  });

  test("float options take Python's float() syntax", () => {
    const timeout: OptionSpec = { name: "timeout", type: "float", metavar: "FLOAT", help: "Seconds." };
    const spec: CommandSpec = { name: "models", summary: "Models.", positionals: [], options: [timeout] };
    const value = (text: string) => (parseCommand(spec, [`--timeout=${text}`]) as ParsedArgs).options["timeout"];
    const accepted: [string, number][] = [
      ["0.5", 0.5],
      [" 2 ", 2],
      [".5", 0.5],
      ["5.", 5],
      ["+1.5", 1.5],
      ["1_0.5", 10.5],
      ["1e3", 1000],
      ["1E3", 1000],
      ["1e1_0", 1e10],
      ["\t.5\n", 0.5],
      ["-1", -1],
      ["inf", Infinity],
      ["-Infinity", -Infinity],
      ["nan", NaN],
      ["NaN", NaN],
      ["+nan", NaN],
    ];
    for (const [text, expected] of accepted) expect(value(text)).toBe(expected);
    expect(parseCommand(spec, ["--timeout", "8"])).toEqual({ help: false, args: {}, options: { timeout: 8 } });
    expect(parseCommand(spec, [])).toEqual({ help: false, args: {}, options: {} });
    for (const text of ["x", "", "1__0", "_1", "1_", "1_.5", "1._5", "1e", "0x10", "infinit", "\x1c1"]) {
      expect(usage(spec, [`--timeout=${text}`])).toBe(`Invalid value for '--timeout': ${reprText(text)} is not a valid float.`);
    }
    expect(usage(spec, ["--timeout", "x"])).toBe("Invalid value for '--timeout': 'x' is not a valid float.");
    expect(helpText(spec)).toContain("  --timeout FLOAT  Seconds.\n");
    const { metavar: _, ...bare } = timeout;
    expect(helpText({ ...spec, options: [bare] })).toContain("  --timeout FLOAT  Seconds.\n");
  });

  test("variadic positionals, and -- ends the options", () => {
    expect(parseCommand(CHECK, ["a.yaml", "b.yaml", "--", "--json"])).toEqual({
      help: false,
      args: { flows: ["a.yaml", "b.yaml", "--json"] },
      options: { json: false },
    });
  });

  test("an optional positional, given and omitted", () => {
    const auth: CommandSpec = {
      name: "auth",
      summary: "Saved logins.",
      positionals: [{ name: "provider", help: "Provider.", optional: true }],
      options: [],
    };
    expect(parseCommand(auth, ["openai"])).toEqual({ help: false, args: { provider: "openai" }, options: {} });
    expect(parseCommand(auth, [])).toEqual({ help: false, args: {}, options: {} });
    expect(helpText(auth).startsWith("Usage: yian auth [OPTIONS] [PROVIDER]\n")).toBe(true);
  });

  test("a missing positional", () => {
    expect(usage(RUN, ["-q"])).toBe("Missing argument 'FLOW'.");
    expect(usage(CHECK, ["--json"])).toBe("Missing argument 'FLOWS...'.");
  });

  test("an extra positional", () => {
    expect(usage(RUN, [flow, "x"])).toBe("Got unexpected extra argument(s) (x)");
    expect(usage(RUN, [flow, "x", "y"])).toBe("Got unexpected extra argument(s) (x y)");
  });

  test("an unknown option, a missing value, a value on a boolean", () => {
    expect(usage(RUN, [flow, "--nope"])).toBe("No such option: --nope");
    expect(usage(RUN, [flow, "-x"])).toBe("No such option: -x");
    expect(usage(RUN, [flow, "--input"])).toBe("Option '--input' requires an argument.");
    expect(usage(RUN, [flow, "--json=1"])).toBe("Option '--json' does not take a value.");
    expect(usage(RUN, [flow, "--quiet=1"])).toBe("Option '--quiet' does not take a value.");
  });

  test("--help returns help before any other check", () => {
    expect(parseCommand(RUN, ["--help"])).toEqual({ help: true });
    expect(parseCommand(RUN, ["missing.yaml", "--help"])).toEqual({ help: true });
  });

  test("file checks", () => {
    const missing = join(home, "missing.yaml");
    expect(usage(RUN, [missing])).toBe(`Invalid value for 'FLOW': File '${missing}' does not exist.`);
    expect(usage(RUN, [home])).toBe(`Invalid value for 'FLOW': File '${home}' is a directory.`);
    expect(usage(RUN, [flow, "--log", home])).toBe(`Invalid value for '--log': File '${home}' is a directory.`);
    expect(parseCommand(RUN, [flow, "--log", missing])).toMatchObject({ options: { log: missing } });
    // Declared order: the positional is checked before `--resume`.
    expect(usage(RUN, ["--resume", missing, missing])).toContain("Invalid value for 'FLOW'");
  });

  test.skipIf(process.getuid?.() === 0)("readable and writable checks", () => {
    const locked = join(home, "locked.yaml");
    writeFileSync(locked, "");
    chmodSync(locked, 0o000);
    expect(usage(RUN, [locked])).toBe(`Invalid value for 'FLOW': File '${locked}' is not readable.`);
    const readOnly = join(home, "run.jsonl");
    writeFileSync(readOnly, "");
    chmodSync(readOnly, 0o444);
    expect(usage(RUN, [flow, "--resume", readOnly])).toBe(
      `Invalid value for '--resume': File '${readOnly}' is not writable.`,
    );
  });

  test("directory checks", () => {
    const missing = join(home, "nodir");
    expect(usage(CHECK, ["--root", missing, "x.yaml"])).toBe(`Invalid value for '--root': Directory '${missing}' does not exist.`);
    expect(usage(CHECK, ["--root", flow, "x.yaml"])).toBe(`Invalid value for '--root': Directory '${flow}' is a file.`);
    expect(usage(RUN, [flow, "--memory-root", flow])).toBe(`Invalid value for '--memory-root': Directory '${flow}' is a file.`);
    const memory = join(home, "memory");
    mkdirSync(memory);
    expect(parseCommand(CHECK, ["--root", memory, "x.yaml"])).toMatchObject({ options: { root: memory } });
  });
});

describe("helpText and usageError", () => {
  test("help starts with the usage line and lists every parameter", () => {
    const text = helpText(RUN);
    expect(text.startsWith("Usage: yian run [OPTIONS] FLOW\n")).toBe(true);
    expect(text).toContain(RUN.summary);
    expect(text).toContain("Path to a flow .yaml");
    expect(text).toContain("-w, --num-workers INTEGER");
    expect(text).toContain("--memory-root DIR");
    expect(text).toContain("--help");
    expect(helpText(CHECK).startsWith("Usage: yian check [OPTIONS] FLOWS...\n")).toBe(true);
  });

  test("a usage error names the command, its help and the error", () => {
    expect(usageError(RUN, new UsageError("Missing argument 'FLOW'."))).toBe(
      "Usage: yian run [OPTIONS] FLOW\nTry 'yian run --help' for help.\n\nError: Missing argument 'FLOW'.\n",
    );
  });
});

describe("printJson", () => {
  test("writes format first, the rest key-sorted as a run-log line", async () => {
    const chunks: string[] = [];
    const out: Writer = { write: async (text) => void chunks.push(text) };
    await printJson({ z: 2n, a: 2, inf: Infinity, nested: { b: 1n, a: "x" } }, out);
    expect(chunks).toEqual([
      `{"format":"${JSON_FORMAT}","a":2.0,"inf":{"$float":"inf"},"nested":{"a":"x","b":1},"z":2}\n`,
    ]);
    chunks.length = 0;
    await printJson({}, out);
    expect(chunks).toEqual(['{"format":"yian.cli.v1"}\n']);
    chunks.length = 0;
    // A one-key document isn't escaped: with `format` it has two keys.
    await printJson({ $record: 1n }, out);
    expect(chunks).toEqual(['{"format":"yian.cli.v1","$record":1}\n']);
  });

  test("resolves after the write's callback", async () => {
    let release!: () => void;
    const out: Writer = { write: () => new Promise<void>((resolve) => (release = resolve)) };
    let done = false;
    const printed = printJson({ ok: true }, out).then(() => (done = true));
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(done).toBe(false);
    release();
    await printed;
    expect(done).toBe(true);
  });
});

describe("streamWriter", () => {
  test("flush waits for the last write", async () => {
    const written: string[] = [];
    const callbacks: (() => void)[] = [];
    const stream = new Writable({
      highWaterMark: 1,
      write(chunk, _encoding, callback) {
        written.push(String(chunk));
        callbacks.push(() => callback());
      },
    });
    const writer = streamWriter(stream);
    void writer.write("a");
    void writer.write("b");
    let flushed = false;
    const io = { stdout: writer, stderr: writer, stdin: process.stdin, stderrStream: process.stderr, env: base };
    const done = flush(io).then(() => (flushed = true));
    while (callbacks.length > 0 || written.length < 2) {
      await new Promise((resolve) => setTimeout(resolve, 1));
      expect(flushed).toBe(false);
      callbacks.shift()?.();
    }
    await done;
    expect(written).toEqual(["a", "b"]);
  });

  test("a write to a closed pipe resolves quietly", async () => {
    const stream = new Writable({
      write(_chunk, _encoding, callback) {
        callback(Object.assign(new Error("write EPIPE"), { code: "EPIPE" }));
      },
    });
    const writer = streamWriter(stream);
    await writer.write("lost");
    await writer.write("after");
  });

  test("another write error rejects", async () => {
    const stream = new Writable({
      write(_chunk, _encoding, callback) {
        callback(Object.assign(new Error("write EIO"), { code: "EIO" }));
      },
    });
    const error = await rejection(streamWriter(stream).write("x"));
    expect(error.message).toBe("write EIO");
  });
});
