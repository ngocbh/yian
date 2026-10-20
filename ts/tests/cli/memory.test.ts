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
 * Later (T7-1 PR 2, need `yian run`): every other row (exit codes, ordering before
 * inputs/provider/log, resume, cleanup reporting, controller drain, pooled cancellation, help text,
 * module docstrings, `test_resume_accepts_one_def_called_from_two_nodes`), and
 * `tests/cli/test_memory_persistence.py` (needs `runFlow`, the AGENT loop and the run's memory
 * wiring).
 *
 * Not ported: `test_no_memory_flow_imports_no_concrete_backend` (TS imports the backends
 * statically; `platform.ts` loads the SQLite driver only on first open).
 *
 * TS-only: the parsed storage is a null-prototype record that keeps a `__proto__` key as data; an
 * undeclared key's message names the declared keys, or `none`; `env` reaches the built memories;
 * with `YIAN_TEST_NODE` set, the built `buildCliMemories` runs on Node with a root, opens the
 * episodic memory and closes the registry, and Bun then sees `<root>/<flow id>/notes.sqlite`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { buildCliMemories, MemoryOptionError, parseMemoryStorage } from "../../src/cli/memory.ts";
import { CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { DeclSite, MemoryDecl, MemorySetupError } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

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
  await using registry = await buildCliMemories(codeFlow([]), { root: undefined, storageOptions: [] });
  expect(registry.size).toBe(0);
});

test("env reaches the built memories", async () => {
  const env = (_name: string): string | undefined => undefined;
  const flow = codeFlow([decl("lessons", "reflective")]);
  await using registry = await buildCliMemories(flow, { root: undefined, storageOptions: [], env });
  expect((registry.get("lessons") as unknown as { env: unknown }).env).toBe(env);
});

test.each([
  ["episodic", { max_episodes: 4n, recall_limit: 2n }],
  ["reflective", { max_items: 5n, recall_limit: 3n }],
] as const)("YAML config reaches the %s constructor", async (type, config) => {
  const flow = codeFlow([decl(type, type, config)]);
  await using registry = await buildCliMemories(flow, { root: undefined, storageOptions: [], env: modelEnv() });
  const session = await registry.get(type)!.openRun();
  try {
    expect(session.snapshot.config).toEqual(config);
  } finally {
    await session.close();
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
    await using registry = await buildCliMemories(flow(), { ...options, env: modelEnv() });
    expect([...registry.keys()]).toEqual([key]);
    const session = await registry.get(key)!.openRun();
    await session.close();
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
const ROOT = join(import.meta.dir, "../..");
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
