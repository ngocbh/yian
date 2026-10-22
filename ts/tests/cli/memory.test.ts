/**
 * `yian run`'s memory flags (`cli/memory.ts`): a port of `tests/cli/test_memory.py`.
 *
 * Ported (direct): `test_memory_storage_key_is_stripped_and_path_kept`,
 * `test_memory_root_without_flow_id_is_a_bad_parameter`, `test_no_declarations_build_an_empty_registry`
 * (Python's `_builtin` patch dropped: the row checks `size === 0`) and
 * `test_yaml_config_reaches_the_built_in_constructor` (both rows: `buildCliMemories`, `openRun()`,
 * the session's `snapshot.config` equals the YAML config).
 *
 * Ported (the builder half of a CLI row, called directly):
 * `test_bad_memory_storage_is_a_bad_parameter_before_inputs_provider_and_log` (5 option rows:
 * `MemoryOptionError`, `paramHint` `--memory-storage`, the message doesn't contain the flag),
 * `test_bad_storage_path_shape_fails_before_inputs_provider_and_log` (`MemorySetupError` `memory
 * 'notes': storage ':memory:' is not a local file path`),
 * `test_unsafe_flow_id_with_memory_root_names_the_memory_root` (the exact message `flow id 'a/b'
 * cannot name a directory under the memory root`; the root dir not created),
 * `test_memory_storage_for_a_custom_key_is_a_bad_parameter`, `test_custom_type_points_to_the_python_api`
 * (fresh half: `MemorySetupError` naming `buildMemories`) and
 * `test_cli_activates_declared_memory_without_learning` (construction half: each built-in type from
 * `--memory-root`/`--memory-storage` gets the derived or given path, seen after `openRun()` creates
 * the file).
 *
 * Changed:
 * - The custom-type message names the TS API, not "the Python API".
 * - `storage_options` is `storageOptions`.
 * - Messages write lists in the text form (`["vec"]`) and quote a lone key as Python's `repr` does.
 * - Flows are built by hand (`CompiledFlow` with START, a CODE or AGENT node and END) instead of
 *   loaded from YAML; the YAML config's ints are `bigint`s, as the loader gives them.
 *
 * Ported through `loadFlow` (Python's `_configured` text): `test_yaml_config_outside_the_allowlist_fails_at_load`,
 * `test_rejected_yaml_config_is_a_load_error_naming_the_key` (the cause's class is TS's: `RangeError`
 * for Python's `ValueError`).
 *
 * Ported through `main` (`yian run`, the "yian run" block):
 * `test_memory_root_persists_every_builtin_under_the_flow_id`,
 * `test_memory_storage_is_keyed_by_memory_key`, `test_no_flags_keep_builtins_in_ram`, the CLI halves
 * of `test_bad_memory_storage_is_a_bad_parameter_before_inputs_provider_and_log` (5 rows),
 * `test_bad_storage_path_shape_fails_before_inputs_provider_and_log`,
 * `test_unsafe_flow_id_with_memory_root_names_the_memory_root`,
 * `test_memory_storage_for_a_custom_key_is_a_bad_parameter` and
 * `test_custom_type_points_to_the_python_api` (fresh row), and
 * `test_cli_activates_declared_memory_without_learning` (6 rows),
 * `test_cli_cleanup_failure_has_concise_host_error`,
 * `test_open_failure_closes_every_backend_and_creates_no_directory`,
 * `test_cli_preserves_primary_when_cleanup_also_fails` and `test_run_help_describes_memory_flags`.
 * Changed there:
 * - Python's patched `model_from_config` is a fake `fetch` answering as an OpenAI-compatible chat
 *   endpoint (xAI's route; its key and the default provider and model in the command's env).
 * - "Before inputs and provider": instead of patching the input gate to fail, the flow declares a
 *   required input no flag supplies, so reaching the gate would be another usage error.
 * - "No Traceback" is no stack line (`    at `).
 * - Python's `_wrap_builtins` close wrappers patch `EpisodicMemory.prototype.close`; the
 *   cleanup-also-fails row gets its initiating failure from the run itself (a TOOL node naming a
 *   tool the run lacks) instead of a patched `_run_flow_controlled`, on episodic memory instead of
 *   a fake session backend.
 * - No-flags-in-RAM ignores the run log's own `run.jsonl.lock`.
 *
 * Ported through `main` (`yian run --resume`, the "yian run --resume" block):
 * `test_custom_type_points_to_the_python_api` (resume row), `test_flags_are_accepted_with_resume`,
 * `test_resume_opens_memory_only_when_the_record_needs_it` (4 rows) and
 * `test_resume_accepts_one_def_called_from_two_nodes`. Changed there: the recorded runs are
 * `runFlow` calls with the offline model and the command's env; the opens are counted on
 * `SessionMemory.prototype.openRun` (Python wraps the built-in constructors), the holder opening
 * its file before the count starts; Python's patched key checks aren't needed, xAI's key being set.
 *
 * Not ported: `test_no_memory_flow_imports_no_concrete_backend` (TS imports the backends
 * statically; `platform.ts` loads the SQLite driver only on first open);
 * `test_first_open_and_cleanup_run_in_controller_after_worker_drain` and
 * `test_real_pooled_cancellation_keeps_storage_until_model_callback_drains` (Python's controller
 * thread and terminal broker; TS has no threads); `test_cli_module_docstrings_name_the_new_flags`
 * (Python docstrings).
 *
 * TS-only: the parsed storage is a null-prototype record that keeps a `__proto__` key as data; an
 * undeclared key's message names the declared keys, or `none`; `env` reaches the built memories;
 * with `YIAN_TEST_NODE` set, the built `buildCliMemories` runs on Node with a root, opens the
 * episodic memory and closes the registry, and Bun then sees `<root>/<flow id>/notes.sqlite`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { pathToFileURL } from "node:url";
import { inspect } from "node:util";

import { main } from "../../src/cli/index.ts";
import { buildCliMemories, MemoryOptionError, parseMemoryStorage } from "../../src/cli/memory.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent, RunStarted } from "../../src/events.ts";
import { DeclSite, MemoryDecl, MemoryOperationError, MemorySetupError } from "../../src/memory/contracts.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { buildMemories } from "../../src/memory_host.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";
import { NO_ENV } from "../engine/run_state_fixtures.ts";

/** The runtime's own formatter, the one `console.log` uses: Bun's on Bun, `node:util`'s on Node. */
const show = typeof Bun === "undefined" ? inspect : Bun.inspect;

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-memory-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** One flow-level declaration at the root flow's site. */
function decl(name: string, type: string, config: Record<string, unknown> = {}): MemoryDecl {
  return new MemoryDecl(name, type, config, new DeclSite([], ["root"]));
}

/** START, one body node and END. */
function graph(body: Node): Map<string, Node> {
  return new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [] })],
    [body.id, body],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
}

/** A flow declaring `memories` around one CODE node (Python's `_flow_text`). */
function codeFlow(memories: readonly MemoryDecl[], flowId: string | null = "cli-memory"): ComposedFlow {
  return new ComposedFlow(
    new CompiledFlow(graph(new CodeNode("work", { code: "return 7" })), [], { flowId, memories: [...memories] }),
  );
}

/** A flow whose AGENT node declares `chat` as session memory (Python's `_SESSION_FLOW`). */
function sessionFlow(): ComposedFlow {
  const chat = new MemoryDecl("chat", "session", {}, new DeclSite([], ["root"], "work"));
  const body = new AgentNode("work", { prompt: "hello", memories: [chat] });
  return new ComposedFlow(new CompiledFlow(graph(body), [], { flowId: "cli-memory", memories: [] }));
}

/** An `Env` with a default model, so reflective memory opens, and no auth file. */
function modelEnv(): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", join(tmp, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
  return (name) => vars.get(name);
}

/** Return what `fn` throws or rejects with, failing when it succeeds. */
async function caught(fn: () => unknown): Promise<unknown> {
  try {
    await fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected an error");
}

/** Expect `fn` to fail with a `MemoryOptionError` for `paramHint`, and return it. */
async function optionError(fn: () => unknown, paramHint: string): Promise<MemoryOptionError> {
  const error = await caught(fn);
  expect(error).toBeInstanceOf(MemoryOptionError);
  expect((error as MemoryOptionError).paramHint).toBe(paramHint);
  return error as MemoryOptionError;
}

/** Expect `fn` to fail with a `MemorySetupError`, and return its message. */
async function setupMessage(fn: () => unknown): Promise<string> {
  const error = await caught(fn);
  expect(error).toBeInstanceOf(MemorySetupError);
  return (error as Error).message;
}

test.each([
  [["notes"], "expected KEY=PATH, got 'notes'"],
  [["notes="], "expected KEY=PATH, got 'notes='"],
  [["=x.sqlite"], "expected KEY=PATH, got '=x.sqlite'"],
  [["missing=a.sqlite"], "key 'missing' is not a memory declared by this flow (declared: notes)"],
  [["notes=a.sqlite", "notes=b.sqlite"], "duplicate key 'notes'"],
] as const)("bad memory storage %p is a bad parameter", async (options, message) => {
  const flow = codeFlow([decl("notes", "episodic")]);
  const error = await optionError(
    () => buildCliMemories(flow, { root: undefined, storageOptions: options }),
    "--memory-storage",
  );
  expect(error.message).toBe(message);
  // The CLI prefixes the flag name; the message itself does not repeat it.
  expect(error.message).not.toContain("--memory-storage");
});

test("a key not declared names none when the flow declares nothing", async () => {
  const error = await optionError(
    () => buildCliMemories(codeFlow([]), { root: undefined, storageOptions: ["notes=a.sqlite"] }),
    "--memory-storage",
  );
  expect(error.message).toBe("key 'notes' is not a memory declared by this flow (declared: none)");
});

test("memory storage key is stripped and path kept", () => {
  const paths = parseMemoryStorage([" notes =./a b.sqlite"]);
  expect(Object.getPrototypeOf(paths)).toBeNull();
  expect(Object.entries(paths)).toEqual([["notes", "./a b.sqlite"]]);
  expect(() => parseMemoryStorage(["notes=a", " notes=b"])).toThrow("duplicate key 'notes'");
});

test("a __proto__ key is kept as data", () => {
  const paths = parseMemoryStorage(["__proto__=p.sqlite"]);
  expect(Object.keys(paths)).toEqual(["__proto__"]);
  expect(Object.getOwnPropertyDescriptor(paths, "__proto__")?.value).toBe("p.sqlite");
});

test("bad storage path shape is a setup error", async () => {
  const flow = codeFlow([decl("notes", "episodic")]);
  const message = await setupMessage(() =>
    buildCliMemories(flow, { root: undefined, storageOptions: ["notes=:memory:"] }),
  );
  expect(message).toBe("memory 'notes': storage ':memory:' is not a local file path");
});

test("unsafe flow id with memory root names the memory root", async () => {
  const root = join(tmp, "mem");
  const flow = codeFlow([decl("notes", "episodic")], "a/b");
  const message = await setupMessage(() => buildCliMemories(flow, { root, storageOptions: [] }));
  expect(message).toBe("flow id 'a/b' cannot name a directory under the memory root");
  expect(message).not.toContain("root=");
  expect(existsSync(root)).toBe(false);
});

test("memory storage for a custom key is a bad parameter", async () => {
  const flow = codeFlow([decl("vec", "vector")]);
  const error = await optionError(
    () => buildCliMemories(flow, { root: undefined, storageOptions: ["vec=v.sqlite"] }),
    "--memory-storage",
  );
  expect(error.message).toBe(
    "key 'vec' has custom type 'vector'; the CLI stores only built-in types (episodic, profile, reflective, session)",
  );
});

test("memory root without flow id is a bad parameter", async () => {
  // YAML requires id:, so only a flow compiled by hand can lack one.
  const flow = codeFlow([decl("notes", "episodic")], null);
  const error = await optionError(
    () => buildCliMemories(flow, { root: join(tmp, "mem"), storageOptions: [] }),
    "--memory-root",
  );
  expect(error.message).toBe("--memory-root needs a flow with an id: field; memory lives under DIR/<flow id>/");
});

test("custom type points to the TS API", async () => {
  const flow = codeFlow([decl("vec", "vector"), decl("notes", "episodic"), decl("idx", "index")]);
  const message = await setupMessage(() => buildCliMemories(flow, { root: undefined, storageOptions: [] }));
  expect(message).toBe(
    'this flow declares custom memory type(s) for ["idx","vec"]; yian run builds only built-in types ' +
      "(episodic, profile, reflective, session). Use the API: buildMemories(flow, { factories }) or " +
      "{ supplied }, then runFlow(flow, inputs, { memory }).",
  );
});

test("no declarations build an empty registry", async () => {
  const registry = await buildCliMemories(codeFlow([]), { root: undefined, storageOptions: [] });
  try {
    expect(registry.size).toBe(0);
  } finally {
    await registry[Symbol.asyncDispose]();
  }
});

test("env reaches the built memories", async () => {
  const env = (_name: string): string | undefined => undefined;
  const flow = codeFlow([decl("lessons", "reflective")]);
  const registry = await buildCliMemories(flow, { root: undefined, storageOptions: [], env });
  try {
    expect((registry.get("lessons") as unknown as { env: unknown }).env).toBe(env);
  } finally {
    await registry[Symbol.asyncDispose]();
  }
});

test.each([
  ["episodic", { max_episodes: 4n, recall_limit: 2n }],
  ["reflective", { max_items: 5n, recall_limit: 3n }],
] as const)("YAML config reaches the %s constructor", async (type, config) => {
  const flow = codeFlow([decl(type, type, config)]);
  const registry = await buildCliMemories(flow, { root: undefined, storageOptions: [], env: modelEnv() });
  try {
    const session = await registry.get(type)!.openRun();
    try {
      expect(session.snapshot.config).toEqual(config);
    } finally {
      await session.close();
    }
  } finally {
    await registry[Symbol.asyncDispose]();
  }
});

/** Return a flow declaring memory `policy` of type `policy` with YAML `config` (Python's `_configured`). */
function configured(policy: string, config: string): string {
  if (policy === "session") {
    return `id: cli-memory
name: cli_memory
nodes:
  work:
    kind: agent
    long_term_memory: {session: {type: session, config: ${config}}}
    prompt: hello
    output: str
output: \${work.output}
`;
  }
  return `id: cli-memory
name: cli_memory
long_term_memory: {${policy}: {type: ${policy}, config: ${config}}}
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: \${work.output}
`;
}

test.each([
  ["episodic", "{bogus: 1}", 'episodic memory config has unknown key(s) ["bogus"]'],
  ["reflective", "{llm_config: {model: x}}", 'unknown key(s) ["llm_config"]'],
  ["session", "{storage: x.sqlite}", "config key 'storage' is not allowed"],
] as const)("YAML config outside the allowlist fails at load (%s)", async (policy, config, detail) => {
  const error = await caught(() => loadFlow(configured(policy, config)));
  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).message).toContain(detail);
});

test.each([
  ["reflective", "{max_items: 0}", "max_items", RangeError],
  ["episodic", "{recall_limit: two}", "recall_limit", TypeError],
  ["session", "{recent_turns: -1}", "recent_turns", RangeError],
] as const)("rejected YAML config is a load error naming the key (%s)", async (policy, config, detail, cause) => {
  const error = await caught(() => loadFlow(configured(policy, config)));
  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).message).toMatch(new RegExp(`long_term_memory '${policy}'.*${detail}`));
  expect(((error as LoadError).cause as Error).constructor).toBe(cause);
});

describe.each(["episodic", "reflective", "session"] as const)("CLI builds %s memory", (type) => {
  const key = type === "session" ? "chat" : "notes";
  const flow = () => (type === "session" ? sessionFlow() : codeFlow([decl("notes", type)]));

  /** Build with `options`, open and close one session of `key`, and close the registry. */
  async function openOnce(options: { root: string | undefined; storageOptions: readonly string[] }) {
    const registry = await buildCliMemories(flow(), { ...options, env: modelEnv() });
    try {
      expect([...registry.keys()]).toEqual([key]);
      const session = await registry.get(key)!.openRun();
      await session.close();
    } finally {
      await registry[Symbol.asyncDispose]();
    }
  }

  test("at the given --memory-storage path", async () => {
    const path = join(tmp, "memory.sqlite");
    await openOnce({ root: undefined, storageOptions: [`${key}=${path}`] });
    expect(existsSync(path)).toBe(true);
  });

  test("at the path --memory-root derives", async () => {
    await openOnce({ root: join(tmp, "mem"), storageOptions: [] });
    expect(existsSync(join(tmp, "mem", "cli-memory", `${key}.sqlite`))).toBe(true);
  });

  test("in RAM without flags", async () => {
    await openOnce({ root: undefined, storageOptions: [] });
    expect(readdirSync(tmp)).toEqual([]);
  });
});

/** The `ts/` directory. */
const ROOT = join(import.meta.dirname, "../..");
const BUILT = join(ROOT, "dist/src/cli/memory.js");

/** This process's environment without `YIAN_*` keys, so a child reads no ambient yian settings. */
function childEnv(): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("YIAN_")));
}

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/profile_memory.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

describe.skipIf(!NODE)("the built module on Node", () => {
  test("Node builds a rooted episodic memory, opens it and closes the registry", () => {
    const root = join(tmp, "mem");
    const load = (name: string) => JSON.stringify(pathToFileURL(join(ROOT, `dist/src/${name}.js`)).href);
    const script = `
const { buildCliMemories } = await import(${load("cli/memory")});
const { CompiledFlow, END_ID, START_ID } = await import(${load("compile/model")});
const { ComposedFlow } = await import(${load("compose/build/pipeline")});
const { DeclSite, MemoryDecl } = await import(${load("memory/contracts")});
const { CodeNode } = await import(${load("nodes/code/node")});
const { EndNode } = await import(${load("nodes/end/node")});
const { StartNode } = await import(${load("nodes/start/node")});
const nodes = new Map([
  [START_ID, new StartNode(START_ID, { params: [] })],
  ["work", new CodeNode("work", { code: "return 7" })],
  [END_ID, EndNode.value(END_ID, { outputName: null })],
]);
const memories = [new MemoryDecl("notes", "episodic", {}, new DeclSite([], ["root"]))];
const flow = new ComposedFlow(new CompiledFlow(nodes, [], { flowId: "cli-memory", memories }));
const registry = await buildCliMemories(flow, { root: ${JSON.stringify(root)}, storageOptions: [] });
try {
  const session = await registry.get("notes").openRun();
  await session.close();
} finally {
  await registry.close();
}
console.log("ok");
`;
    const result = spawnSync(NODE!, ["--input-type=module", "-e", script], {
      encoding: "utf8",
      timeout: 20_000,
      env: childEnv(),
    });
    expect(result.stderr).toBe("");
    expect(result.stdout).toBe("ok\n");
    expect(result.status).toBe(0);
    expect(existsSync(join(root, "cli-memory", "notes.sqlite"))).toBe(true);
  });
});

// --- `yian run` with the memory flags --- //

/** Python's `_CODE_FLOW`: `memory` declared at flow level (none when `null`) around a CODE node. */
function codeFlowText(memory: string | null = "{notes: episodic}", flowId = "cli-memory"): string {
  const declaration = memory === null ? "" : `long_term_memory: ${memory}\n`;
  return (
    `id: ${flowId}\nname: cli_memory\n${declaration}` +
    "nodes:\n  work:\n    kind: code\n    code: return 7\n    output: int\noutput: ${work.output}\n"
  );
}

/** Python's `_SESSION_FLOW`: an AGENT node with `chat` session memory. */
const SESSION_FLOW = `id: cli-memory
name: cli_memory
nodes:
  work:
    kind: agent
    long_term_memory: {chat: session}
    prompt: hello
    output: str
output: \${work.output}
`;

/** A one-word ("7") chat-completions stream. */
const SEVEN = [
  { choices: [{ index: 0, delta: { role: "assistant", content: "7" }, finish_reason: null }] },
  { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
]
  .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
  .join("") + "data: [DONE]\n\n";

/** Answer every model call with `SEVEN`, so AGENT flows run without a provider. */
async function offlineModel(): Promise<Response> {
  return new Response(SEVEN, { headers: { "content-type": "text/event-stream" } });
}

/**
 * Return flow `text` with a required input no flag supplies, so a run reaching the input gate fails
 * there with another usage error (Python patches the gate to fail instead).
 */
function withRequiredInput(text: string): string {
  return text.replace("nodes:\n", "input:\n  topic: !str\nnodes:\n");
}

/** Write `text` to `tmp/flow.yaml` and return its path. */
function flowFile(text: string): string {
  const path = join(tmp, "flow.yaml");
  writeFileSync(path, text);
  return path;
}

/**
 * Run `yian run <flow> --log tmp/run.jsonl --quiet ...extra` (Python's `_run`) with an offline
 * model, the default provider and model set so reflective memory and AGENT nodes resolve one.
 */
async function runCli(flow: string, ...extra: string[]) {
  const log = join(tmp, "run.jsonl");
  return { ...(await cli([flow, "--log", log, "--quiet", ...extra])), log };
}

/** Run `yian run <flow> --resume <log> --quiet ...extra` as `runCli` runs a fresh one. */
function resumeCli(flow: string, log: string, ...extra: string[]) {
  return cli([flow, "--resume", log, "--quiet", ...extra]);
}

/** The command's env for `runCli` and `resumeCli`. */
const CLI_VARS = (): Record<string, string> => ({
  HOME: join(tmp, "home"),
  XAI_API_KEY: "test-xai-key",
  YIAN_DEFAULT_PROVIDER: "xai",
  YIAN_DEFAULT_MODEL: "grok-4.3",
});

/** `CLI_VARS` as an `Env`. */
function cliEnv(): Env {
  const vars = CLI_VARS();
  return (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined);
}

/** Run `yian run <args>` with an offline model; return its code and output. */
async function cli(args: readonly string[]) {
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: cliEnv(),
  };
  const send = globalThis.fetch;
  globalThis.fetch = offlineModel as unknown as typeof fetch;
  try {
    const code = await main(["run", ...args], io);
    return { code, stdout: out.join(""), stderr: err.join("") };
  } finally {
    globalThis.fetch = send;
  }
}

/** Collapse whitespace, so wrapped messages match (Python's `_plain`; output here has no styles). */
function plain(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(" ");
}

/** Whether `text` holds a stack trace line. */
function hasStack(text: string): boolean {
  return /^\s+at /m.test(text);
}

/** The records of the run log at `path`. */
function records(path: string) {
  return readRunLog(path).map((envelope) => envelope.record);
}

/**
 * Patch `close` of every `EpisodicMemory` to run the real close, then `after` (Python's
 * `_wrap_builtins` close wrappers); returns the restore.
 */
function wrapEpisodicClose(after: () => void): () => void {
  const close = EpisodicMemory.prototype.close;
  EpisodicMemory.prototype.close = function (this: EpisodicMemory) {
    close.call(this);
    after();
  };
  return () => {
    EpisodicMemory.prototype.close = close;
  };
}

describe("yian run", () => {
  beforeEach(() => {
    mkdirSync(join(tmp, "home"));
  });

  test("memory root persists every builtin under the flow id", async () => {
    const root = join(tmp, "mem");
    const flow = flowFile(codeFlowText("{notes: episodic, lessons: reflective, about: profile}"));
    const result = await runCli(flow, "--memory-root", root);
    expect(result.code, result.stderr).toBe(0);
    expect(statSync(join(root, "cli-memory", "notes.sqlite")).isFile()).toBe(true);
    expect(statSync(join(root, "cli-memory", "lessons.sqlite")).isFile()).toBe(true);
    expect(existsSync(join(root, "cli-memory", ".about.md.lock"))).toBe(true);
    expect(records(result.log)[0]).toBeInstanceOf(RunStartedRecord);
  });

  test("memory storage is keyed by memory key", async () => {
    const path = join(tmp, "notes.sqlite");
    const result = await runCli(flowFile(codeFlowText()), "--memory-storage", `notes=${path}`);
    expect(result.code, result.stderr).toBe(0);
    expect(statSync(path).isFile()).toBe(true);
  });

  test("no flags keep builtins in RAM", async () => {
    const flow = flowFile(codeFlowText("{notes: episodic, lessons: reflective, about: profile}"));
    const result = await runCli(flow);
    expect(result.code, result.stderr).toBe(0);
    // The run log's own lock file aside.
    const stored = readdirSync(tmp, { recursive: true, encoding: "utf8" }).filter(
      (name) => /\.(sqlite|md|lock)$/.test(name) && name !== "run.jsonl.lock",
    );
    expect(stored).toEqual([]);
  });

  test.each([
    [["notes"]],
    [["notes="]],
    [["=x.sqlite"]],
    [["missing=a.sqlite"]],
    [["notes=a.sqlite", "notes=b.sqlite"]],
  ] as const)("bad memory storage %p is a bad parameter before inputs, provider and log", async (options) => {
    const flow = flowFile(withRequiredInput(codeFlowText()));
    const result = await runCli(flow, ...options.flatMap((option) => ["--memory-storage", option]));
    expect(result.code, result.stderr).toBe(2);
    // The CLI prefixes the flag name; the message itself does not repeat it.
    expect(plain(result.stderr).split("--memory-storage").length - 1).toBe(1);
    expect(result.stderr).toContain("Error: Invalid value for --memory-storage: ");
    expect(existsSync(result.log)).toBe(false);
  });

  test("bad storage path shape fails before inputs, provider and log", async () => {
    const flow = flowFile(withRequiredInput(codeFlowText()));
    const result = await runCli(flow, "--memory-storage", "notes=:memory:");
    expect(result.code, result.stderr).toBe(1);
    expect(plain(result.stderr)).toContain("memory 'notes': storage ':memory:' is not a local file path");
    expect(hasStack(result.stderr)).toBe(false);
    expect(existsSync(result.log)).toBe(false);
  });

  test("unsafe flow id with memory root names the memory root", async () => {
    const root = join(tmp, "mem");
    const result = await runCli(flowFile(codeFlowText("{notes: episodic}", '"a/b"')), "--memory-root", root);
    expect(result.code, result.stderr).toBe(1);
    const text = plain(result.stderr);
    expect(text).toContain("flow id 'a/b' cannot name a directory under the memory root");
    expect(text).not.toContain("root=");
    expect(existsSync(root)).toBe(false);
    expect(existsSync(result.log)).toBe(false);
  });

  test("memory storage for a custom key is a bad parameter", async () => {
    const result = await runCli(flowFile(codeFlowText("{vec: vector}")), "--memory-storage", "vec=v.sqlite");
    expect(result.code).toBe(2);
    expect(plain(result.stderr)).toContain("custom");
  });

  test("custom type points to the TS API (fresh)", async () => {
    const flow = flowFile(withRequiredInput(codeFlowText("{vec: vector}")));
    const result = await runCli(flow);
    expect(result.code).toBe(1);
    const text = plain(result.stderr);
    expect(text.startsWith("run failed:")).toBe(true);
    expect(text).toContain("Use the API");
    expect(text).toContain("buildMemories");
    expect(hasStack(result.stderr)).toBe(false);
    expect(existsSync(result.log)).toBe(false);
  });

  describe.each(["episodic", "reflective", "session"] as const)("activates declared %s memory", (type) => {
    test.each([false, true])("without learning (durable %p)", async (durable) => {
      const key = type === "session" ? "chat" : "notes";
      const flow = flowFile(type === "session" ? SESSION_FLOW : codeFlowText(`{notes: ${type}}`));
      const path = join(tmp, "memory.sqlite");
      const options = durable ? [`${key}=${path}`] : [];
      const result = await runCli(flow, ...options.flatMap((option) => ["--memory-storage", option]));
      expect(result.code, result.stderr).toBe(0);
      const logged = records(result.log);
      const started = logged[0] as RunStartedRecord;
      expect(started).toBeInstanceOf(RunStartedRecord);
      expect(started.memories!.map((spec) => spec.key)).toEqual([key]);
      expect(started.memorySnapshots!.length).toBe(1);
      expect(started.learnable).toBe(false);
      expect(logged.some((record) => record.constructor.name.startsWith("Backward"))).toBe(false);
      expect(existsSync(path)).toBe(durable);
      if (durable) {
        // The CLI must release the file before returning.
        const loaded = await loadFlow(readFileSync(flow, "utf8"));
        const reopening = { root: undefined, storageOptions: options, env: modelEnv() };
        const registry = await buildCliMemories(loaded, reopening);
        try {
          const session = await registry.get(key)!.openRun();
          try {
            expect(show(session.snapshot)).not.toContain(path);
          } finally {
            await session.close();
          }
        } finally {
          await registry[Symbol.asyncDispose]();
        }
      }
    });
  });

  test("cli cleanup failure has concise host error", async () => {
    const restore = wrapEpisodicClose(() => {
      throw new MemoryOperationError("close", "test.close", "memory close failed", false);
    });
    try {
      const result = await runCli(flowFile(codeFlowText()));
      expect(result.code).toBe(1);
      expect(plain(result.stderr)).toContain("run failed: memory close failed");
      expect(hasStack(result.stderr)).toBe(false);
    } finally {
      restore();
    }
  });

  test("open failure closes every backend and creates no directory", async () => {
    const flow = flowFile(
      SESSION_FLOW.replace("name: cli_memory\n", "name: cli_memory\nlong_term_memory: {archive: episodic}\n"),
    );
    // Run setup opens memories in sorted key order: archive opens (and creates its file), then
    // chat fails because its parent directory is missing.
    const archivePath = join(tmp, "archive.sqlite");
    const absent = join(tmp, "absent");
    const result = await runCli(
      flow,
      "--memory-storage",
      `archive=${archivePath}`,
      "--memory-storage",
      `chat=${join(absent, "chat.sqlite")}`,
    );
    expect(result.code).toBe(1);
    expect(plain(result.stderr)).toContain("run failed:");
    expect(hasStack(result.stderr)).toBe(false);
    expect(existsSync(absent)).toBe(false);
    expect(statSync(archivePath).isFile()).toBe(true); // archive really opened before chat failed
    // Opening takes the store; it would be busy if the run still held it.
    const archive = new EpisodicMemory({ storage: archivePath });
    try {
      archive.openRun().close();
    } finally {
      archive.close();
    }
  });

  test("cli preserves primary when cleanup also fails", async () => {
    let closes = 0;
    const restore = wrapEpisodicClose(() => {
      closes += 1;
      throw new Error("cleanup failure");
    });
    // A TOOL node naming a tool the run lacks: the run rejects after the registry is built.
    const work = "    kind: code\n    code: return 7\n    output: int\n";
    const flow = flowFile(codeFlowText().replace(work, "    kind: tool\n    tool_id: cli_missing_tool\n"));
    try {
      const result = await runCli(flow);
      expect(result.code).toBe(1);
      expect(plain(result.stderr)).toContain("run failed: ");
      expect(plain(result.stderr)).toContain("cli_missing_tool");
      expect(plain(result.stderr)).not.toContain("cleanup failure");
      expect(closes).toBe(1);
    } finally {
      restore();
    }
  });

  test("run help describes memory flags", async () => {
    const out: string[] = [];
    const io: CliIo = {
      stdout: { write: async (text) => void out.push(text) },
      stderr: { write: async (text) => void out.push(text) },
      stdin: new PassThrough(),
      stderrStream: new PassThrough(),
      env: NO_ENV,
    };
    expect(await main(["run", "--help"], io)).toBe(0);
    const text = plain(out.join(""));
    expect(text).toContain("--memory-root");
    expect(text).toContain("KEY=PATH");
    expect(text).toContain("Without either flag, memory lives in process RAM.");
    expect(text).not.toContain("POLICY");
  });
});

describe("yian run --resume", () => {
  beforeEach(() => {
    mkdirSync(join(tmp, "home"));
  });

  test("custom type points to the TS API (resume)", async () => {
    const finished = await runCli(flowFile(codeFlowText(null)));
    expect(finished.code, finished.stderr).toBe(0);
    const before = readFileSync(finished.log);
    const result = await resumeCli(flowFile(codeFlowText("{vec: vector}")), finished.log);
    expect(result.code).toBe(1);
    const text = plain(result.stderr);
    expect(text.startsWith("resume failed:"), text).toBe(true);
    expect(text).toContain("Use the API");
    expect(text).toContain("buildMemories");
    expect(hasStack(result.stderr)).toBe(false);
    expect(readFileSync(finished.log)).toEqual(before);
  });

  test("flags are accepted with resume", async () => {
    const root = join(tmp, "mem");
    const result = await runCli(flowFile(codeFlowText()), "--memory-root", root);
    expect(result.code, result.stderr).toBe(0);
    const before = readFileSync(result.log);
    const resumed = await resumeCli(flowFile(codeFlowText()), result.log, "--memory-root", root);
    expect(resumed.code, resumed.stderr).toBe(0);
    expect(readFileSync(result.log)).toEqual(before);
  });

  describe.each([false, true])("resume opens memory only when the record needs it (terminal %p)", (terminal) => {
    test.each([false, true])("enabled %p", async (enabled) => {
      const flow = flowFile(SESSION_FLOW);
      const log = join(tmp, "run.jsonl");
      const memoryPath = join(tmp, "memory.sqlite");
      const loaded = await loadFlow(SESSION_FLOW);
      const primary = new Error("stop before work");
      const stop = (event: RunEvent) => {
        if (!terminal && event instanceof NodeStarted && event.nodeId === "work") throw primary;
      };
      const backend = new SessionMemory();
      const send = globalThis.fetch;
      globalThis.fetch = offlineModel as unknown as typeof fetch;
      try {
        const memory = enabled ? new Map([["chat", backend]]) : null;
        const options = { log, onEvent: stop, env: cliEnv(), memory };
        const running = runFlow(loaded, {}, options);
        if (terminal) expect((await running).status).toBe("succeeded");
        else await expect(running).rejects.toBe(primary);
      } finally {
        globalThis.fetch = send;
        backend.close();
      }
      const before = readFileSync(log);

      const opened: string[] = [];
      // An occupied store must not matter to terminal or historically dormant recovery; opening a
      // run takes the file.
      const holder = terminal || !enabled ? new SessionMemory({ storage: memoryPath }) : null;
      holder?.openRun().close();
      const openRun = SessionMemory.prototype.openRun;
      SessionMemory.prototype.openRun = function (this: SessionMemory, ...args) {
        opened.push("session");
        return openRun.apply(this, args);
      };
      let result;
      try {
        result = await resumeCli(flow, log, "--memory-storage", `chat=${memoryPath}`);
      } finally {
        SessionMemory.prototype.openRun = openRun;
        holder?.close();
      }
      expect(opened).toEqual(terminal || !enabled ? [] : ["session"]);
      expect(result.code, result.stderr).toBe(0);
      const after = readFileSync(log);
      expect(after.subarray(0, before.length)).toEqual(before);
      if (terminal) expect(after).toEqual(before);
    });
  });

  test("resume accepts one def called from two nodes", async () => {
    const text =
      "id: two-calls\nname: two_calls\nlong_term_memory: {notes: episodic}\n" +
      "defs:\n  child:\n    nodes:\n      work: {kind: code, code: return 1, output: int}\n" +
      "    output: ${work.output}\n" +
      "nodes:\n  one: {kind: call, call: child}\n  two: {kind: call, call: child}\n" +
      "output: ${one.output + two.output}\n";
    const flow = flowFile(text);
    const log = join(tmp, "run.jsonl");
    const stopAfterStart = (event: RunEvent) => {
      if (event instanceof RunStarted) throw new Error("host stop");
    };
    const composed = await loadFlow(text);
    const registry = await buildMemories(composed);
    try {
      const run = runFlow(composed, {}, { log, memory: registry, onEvent: stopAfterStart, env: cliEnv() });
      await expect(run).rejects.toThrow("host stop");
      const resumed = await resumeCli(flow, log);
      expect(resumed.code, resumed.stderr).toBe(0);
    } finally {
      await registry[Symbol.asyncDispose]();
    }
  });
});
