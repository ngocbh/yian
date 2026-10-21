/**
 * Host-side construction of the memories a flow declares (`memory_host.ts`): a port of
 * `tests/engine/test_memory_host.py`.
 *
 * Ported: `test_registry_is_a_read_only_mapping_with_types`,
 * `test_close_closes_owned_objects_in_reverse_order_once`,
 * `test_close_keeps_the_first_failure_and_notes_the_rest`,
 * `test_close_retries_only_the_backends_that_failed`, `test_close_copies_the_notes_of_later_failures`,
 * `test_exit_keeps_the_primary_exception`, `test_builtins_are_constructed_from_declared_config`,
 * `test_supplied_wins_and_is_not_closed`, `test_factories_build_custom_types_with_config`,
 * `test_setup_errors` (every case), `test_storage_values_must_be_text_paths` (every case),
 * `test_storage_path_errors_show_the_plain_path`, `test_flow_must_be_a_composed_flow`,
 * `test_setup_errors_carry_the_declaration_origin`, `test_flow_without_memory_builds_an_empty_registry`,
 * `test_root_layout_follows_flow_id_scope_and_name`,
 * `test_explicit_storage_overrides_root_and_needs_an_existing_parent`,
 * `test_two_keys_may_not_share_a_storage_path`, `test_explicit_storage_may_not_reuse_a_root_derived_path`,
 * `test_no_root_means_ram_and_no_flow_id_is_needed`, `test_root_requires_a_safe_flow_id` (every
 * case), `test_unsafe_flow_ids_are_rejected_before_touching_disk` (every case),
 * `test_partial_failure_closes_what_was_built_in_reverse`,
 * `test_factory_result_that_is_not_a_memory_is_closed`,
 * `test_second_registry_on_one_root_is_busy_until_the_first_closes` and
 * `test_open_after_close_is_closed_and_opens_no_file` (every case).
 *
 * Changed:
 * - `buildMemories` and `close` are async. The registry is a read-only `Map` subclass, and so is
 *   its `types`.
 * - `test_exit_keeps_the_primary_exception`: `await using` throws a `SuppressedError` whose `error`
 *   is the cleanup failure (with its `notes`) and whose `suppressed` is the primary (the language's
 *   rule; Python notes the cleanup failure on the primary).
 * - `storage`, `supplied` and `factories` are plain records. A factory gets `toHost(config)` (safe
 *   ints as `number`) and may be async. JS ignores extra arguments, so the "rejected its config" row
 *   uses a factory throwing `RangeError` (Python's `ValueError`).
 * - `test_storage_values_must_be_text_paths`: the `bytes`, `_BytesPath` and `None` rows are a
 *   `Buffer`, a `URL` and `null`; the int row stays a number. The message is
 *   `storage["notes"] must be a string path, not <type>`.
 * - Messages use the TS spellings: lists and quoted keys in the text form, keyword names without
 *   Python's `=`; `caught.value.__notes__` is `error.notes`; Python's `RuntimeError` is an `Error`.
 * - Flows are built by hand (`CompiledFlow` with START, a CODE or AGENT node and END; a `CallNode`
 *   whose child flow `lib` declares `notes` at scope `["lib"]` for the `uses:` rows) instead of
 *   loaded from YAML; flow ids the loader refuses are set with `Object.assign`.
 * - `test_setup_errors_carry_the_declaration_origin` checks the hand-built `OriginRef`: there is no
 *   loader to resolve a source line.
 *
 * TS-only: a `root` of `""` derives `<flow id>/notes.sqlite` relative to the current directory; a
 * `root` that isn't a string is a `TypeError`; `null` records read as empty; a custom type named
 * `toString` with no factory fails
 * "has no factory"; `set`, `delete`, `clear`, `getOrInsert` and `getOrInsertComputed` on the
 * registry and on `types` throw `TypeError`; a host object whose `close()` rejects is awaited and
 * reported like a sync failure; two overlapping `close()` calls close each object once and both
 * reject with the same first failure, and a later call retries it; an async factory is awaited; an
 * async factory rejecting `TypeError` is "rejected its config"; a wrapped failure keeps its cause;
 * `env` reaches reflective and profile memory and defaults to `processEnv`.
 *
 * Ported, through `runFlow` in an environment holding only the test's credential store:
 * `test_registry_built_for_another_flow_is_rejected` and `test_registry_for_this_flow_runs`, whose
 * flows are loaded from YAML.
 * Changed: `test_run_flow_accepts_any_mapping` becomes "a plain object is refused; a Map runs":
 * `memory` must be a `Map`, as `skills` must.
 *
 * Later (T4-8): `test_package_exports_the_host_builder`, `test_resume_checks_registry_types_too`,
 * `test_resume_accepts_one_callee_called_from_two_nodes` (every case) and
 * `test_backward_checks_registry_types_too`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { processEnv } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import {
  DeclSite,
  type Memory,
  MemoryDecl,
  MemoryOperationError,
  MemorySetupError,
} from "../../src/memory/contracts.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { buildMemories, MemoryRegistry } from "../../src/memory_host.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";

import { FakeMemory } from "./memory_fakes.ts";

let tmp: string;

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "yian-memory-host-"));
});

afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

/** A host object that logs its close and throws `error` from it, if set. */
class Closable {
  readonly log: string[];
  readonly name: string;
  error: unknown;

  constructor(log: string[], name: string, error: unknown = null) {
    this.log = log;
    this.name = name;
    this.error = error;
  }

  openRun(): never {
    throw new Error("not opened in these tests");
  }

  reset(): void {}

  close(): void {
    this.log.push(this.name);
    if (this.error !== null) throw this.error;
  }
}

/** A `Closable` whose close fails only the first time. */
class FailsOnce extends Closable {
  override close(): void {
    const error = this.error;
    this.error = null;
    this.log.push(this.name);
    if (error !== null) throw error;
  }
}

/** A `Closable` whose close returns a promise that rejects with `error`, if set. */
class AsyncClosable extends Closable {
  override async close(): Promise<void> {
    await Promise.resolve();
    super.close();
  }
}

/** Return an `Error` carrying one note. */
function failureWithNote(message: string, note: string): Error {
  return Object.assign(new Error(message), { notes: [note] });
}

/** A custom memory that keeps the config its factory got. */
class Custom {
  readonly config: Record<string, unknown>;

  constructor(config: Record<string, unknown> = {}) {
    this.config = config;
  }

  openRun(): never {
    throw new Error("not opened");
  }

  reset(): void {}
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

/** Expect `fn` to fail with a `MemorySetupError` matching `pattern`, and return it. */
async function setupError(fn: () => unknown, pattern?: RegExp): Promise<MemorySetupError> {
  const error = await caught(fn);
  expect(error).toBeInstanceOf(MemorySetupError);
  if (pattern !== undefined) expect((error as Error).message).toMatch(pattern);
  return error as MemorySetupError;
}

/** Return the `code` of the `MemoryOperationError` that `fn` throws or rejects with. */
async function operationCode(fn: () => unknown): Promise<string> {
  const error = await caught(fn);
  expect(error).toBeInstanceOf(MemoryOperationError);
  return (error as MemoryOperationError).code;
}

/** One flow-level declaration at the root flow's site. */
function decl(name: string, type: string, config: Record<string, unknown> = {}, origin: OriginRef | null = null) {
  return new MemoryDecl(name, type, config, new DeclSite([], ["root"]), origin);
}

/** START, one body node and END. */
function graph(body: Node): Map<string, Node> {
  return new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [] })],
    [body.id, body],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
}

/**
 * A flow declaring `memories` around one CODE node, or around an AGENT `ask` declaring
 * `agentMemories` (Python's `_flow`).
 */
function hostFlow(
  memories: readonly MemoryDecl[],
  { flowId = "host-flow", agentMemories }: { flowId?: string | null; agentMemories?: readonly MemoryDecl[] } = {},
): ComposedFlow {
  const body =
    agentMemories === undefined
      ? new CodeNode("echo", { code: "return 1" })
      : new AgentNode("ask", { prompt: "hi", memories: [...agentMemories] });
  return new ComposedFlow(new CompiledFlow(graph(body), [], { flowId, memories: [...memories] }));
}

/** A flow `app` declaring `notes` and `about` that calls a child flow `lib` declaring `notes`. */
function importFlow(): ComposedFlow {
  const child = new CompiledFlow(graph(new CodeNode("a", { code: "return 1" })), [], {
    flowId: "lib",
    memories: [new MemoryDecl("notes", "episodic", {}, new DeclSite(["lib"], ["root"]))],
  });
  return new ComposedFlow(
    new CompiledFlow(graph(new CallNode("a", { flowId: "lib", child })), [], {
      flowId: "app",
      memories: [decl("notes", "episodic"), decl("about", "profile")],
    }),
  );
}

/** Read one private field of a backend, as Python's tests read private attributes. */
function field(memory: Memory | undefined, name: string): unknown {
  return (memory as unknown as Record<string, unknown>)[name];
}

/** Return the names of the entries in `dir`, sorted. */
function entries(dir: string): string[] {
  return fs.readdirSync(dir).sort();
}

test("registry is a read-only mapping with types", () => {
  const log: string[] = [];
  const a = new Closable(log, "a");
  const b = new Closable(log, "b");
  const registry = new MemoryRegistry(
    new Map([
      ["a", a],
      ["b", b],
    ]),
    [
      ["a", "episodic"],
      ["b", "custom"],
    ],
    { owned: [a] },
  );
  expect(registry).toBeInstanceOf(Map);
  expect([...registry]).toEqual([
    ["a", a],
    ["b", b],
  ]);
  expect(Object.fromEntries(registry.types)).toEqual({ a: "episodic", b: "custom" });
  // `Map`'s upserts are outside the ES2024 lib the project compiles against.
  type Upserts = {
    getOrInsert(key: string, value: unknown): unknown;
    getOrInsertComputed(key: string, compute: () => unknown): unknown;
  };
  const writes: ((table: Map<string, unknown>) => unknown)[] = [
    (table) => table.set("a", "x"),
    (table) => table.delete("a"),
    (table) => table.clear(),
    (table) => (table as unknown as Upserts).getOrInsert("z", 1),
    (table) => (table as unknown as Upserts).getOrInsertComputed("z", () => 1),
  ];
  for (const write of writes) {
    expect(() => write(registry.types as Map<string, unknown>)).toThrow(TypeError);
    expect(() => write(registry as Map<string, unknown>)).toThrow("the memory registry and its types are read-only");
  }
  expect(registry.size).toBe(2);
  expect(registry.types.size).toBe(2);
});

test("close closes owned objects in reverse order once", async () => {
  const log: string[] = [];
  const [a, b, c] = ["a", "b", "c"].map((name) => new Closable(log, name)) as [Closable, Closable, Closable];
  const registry = new MemoryRegistry(
    new Map([
      ["a", a],
      ["b", b],
      ["c", c],
    ]),
    ["a", "b", "c"].map((key) => [key, "t"] as const),
    { owned: [a, b] },
  );
  {
    await using held = registry;
    expect(held).toBe(registry);
  }
  expect(log).toEqual(["b", "a"]);
  await registry.close();
  expect(log).toEqual(["b", "a"]);
});

test("close keeps the first failure and notes the rest", async () => {
  const log: string[] = [];
  const first = new Error("first");
  const a = new Closable(log, "a", new Error("second"));
  const b = new Closable(log, "b", first);
  const registry = new MemoryRegistry(
    new Map([
      ["a", a],
      ["b", b],
    ]),
    [
      ["a", "t"],
      ["b", "t"],
    ],
    { owned: [a, b] },
  );
  expect(await caught(() => registry.close())).toBe(first);
  expect(log).toEqual(["b", "a"]);
  expect((first as Error & { notes?: string[] }).notes).toEqual(["Additional memory cleanup failed: Error: second"]);
});

test("close retries only the backends that failed", async () => {
  const log: string[] = [];
  const a = new Closable(log, "a");
  const b = new FailsOnce(log, "b", new Error("busy"));
  const c = new Closable(log, "c");
  const registry = new MemoryRegistry(
    new Map<string, Memory>([
      ["a", a],
      ["b", b],
      ["c", c],
    ]),
    ["a", "b", "c"].map((key) => [key, "t"] as const),
    { owned: [a, b, c] },
  );
  expect(((await caught(() => registry.close())) as Error).message).toBe("busy");
  expect(log).toEqual(["c", "b", "a"]);
  await registry.close();
  expect(log).toEqual(["c", "b", "a", "b"]);
  await registry.close();
  expect(log).toEqual(["c", "b", "a", "b"]);
});

test("close copies the notes of later failures", async () => {
  const log: string[] = [];
  const a = new Closable(log, "a", failureWithNote("second", "second detail"));
  const b = new Closable(log, "b", new Error("first"));
  const registry = new MemoryRegistry(
    new Map([
      ["a", a],
      ["b", b],
    ]),
    [
      ["a", "t"],
      ["b", "t"],
    ],
    { owned: [a, b] },
  );
  const error = (await caught(() => registry.close())) as Error & { notes?: string[] };
  expect(error.message).toBe("first");
  expect(error.notes).toEqual(["Additional memory cleanup failed: Error: second", "second detail"]);
});

test("await using reports the cleanup failure and suppresses the primary", async () => {
  const log: string[] = [];
  const cleanup = failureWithNote("cleanup", "cleanup detail");
  const primary = new RangeError("primary");
  const a = new Closable(log, "a", cleanup);
  const error = await caught(async () => {
    await using _registry = new MemoryRegistry(new Map([["a", a]]), [["a", "t"]], { owned: [a] });
    throw primary;
  });
  // `SuppressedError` is outside the ES2024 lib the project compiles against.
  const Suppressed = (globalThis as unknown as { SuppressedError: new () => Error }).SuppressedError;
  expect(error).toBeInstanceOf(Suppressed);
  expect((error as { error?: unknown }).error).toBe(cleanup);
  expect((error as { suppressed?: unknown }).suppressed).toBe(primary);
  expect((cleanup as Error & { notes?: string[] }).notes).toEqual(["cleanup detail"]);
  expect(Object.hasOwn(primary, "notes")).toBe(false);
});

test("an async close failure is awaited and reported like a sync one", async () => {
  const log: string[] = [];
  const a = new AsyncClosable(log, "a", new Error("second"));
  const b = new AsyncClosable(log, "b", new Error("first"));
  const registry = new MemoryRegistry(
    new Map([
      ["a", a],
      ["b", b],
    ]),
    [
      ["a", "t"],
      ["b", "t"],
    ],
    { owned: [a, b] },
  );
  const error = (await caught(() => registry.close())) as Error & { notes?: string[] };
  expect(error.message).toBe("first");
  expect(error.notes).toEqual(["Additional memory cleanup failed: Error: second"]);
  expect(log).toEqual(["b", "a"]);
});

test("overlapping close calls close each object once and share the first failure", async () => {
  const log: string[] = [];
  const a = new AsyncClosable(log, "a");
  const failure = new Error("busy");
  const b = new AsyncClosable(log, "b", failure);
  const registry = new MemoryRegistry(
    new Map([
      ["a", a],
      ["b", b],
    ]),
    [
      ["a", "t"],
      ["b", "t"],
    ],
    { owned: [a, b] },
  );
  const first = registry.close();
  const second = registry.close();
  expect(second).toBe(first);
  expect(await caught(() => first)).toBe(failure);
  expect(await caught(() => second)).toBe(failure);
  expect(log).toEqual(["b", "a"]);
  b.error = null;
  await registry.close();
  expect(log).toEqual(["b", "a", "b"]);
});

test("builtins are constructed from declared config", async () => {
  const flow = hostFlow(
    [decl("lessons", "reflective"), decl("notes", "episodic", { max_episodes: 5n }), decl("about", "profile")],
    { agentMemories: [new MemoryDecl("chat", "session", { recent_turns: 2n }, new DeclSite([], ["root"], "ask"))] },
  );
  await using registry = await buildMemories(flow);
  expect([...registry.keys()]).toEqual(["about", "chat", "lessons", "notes"]);
  expect(registry.get("notes")!.constructor).toBe(EpisodicMemory);
  expect(registry.get("lessons")!.constructor).toBe(ReflectiveMemory);
  expect(registry.get("chat")!.constructor).toBe(SessionMemory);
  expect(registry.get("about")!.constructor).toBe(ProfileMemory);
  expect(Object.fromEntries(registry.types)).toEqual({
    about: "profile",
    chat: "session",
    lessons: "reflective",
    notes: "episodic",
  });
  // The YAML keys reach the constructors in camelCase.
  expect(field(registry.get("notes"), "requested")).toEqual([5, null]);
  expect(field(registry.get("chat"), "recentTurns")).toBe(2);
});

test("env reaches reflective and profile memory and defaults to processEnv", async () => {
  const flow = hostFlow([decl("lessons", "reflective"), decl("about", "profile")]);
  const env = (_name: string): string | undefined => undefined;
  await using given = await buildMemories(flow, { env });
  expect(field(given.get("lessons"), "env")).toBe(env);
  expect(field(given.get("about"), "env")).toBe(env);
  await using defaulted = await buildMemories(flow);
  expect(field(defaulted.get("lessons"), "env")).toBe(processEnv);
  expect(field(defaulted.get("about"), "env")).toBe(processEnv);
});

test("supplied wins and is not closed", async () => {
  const log: string[] = [];
  const mine = new Closable(log, "mine");
  {
    await using registry = await buildMemories(hostFlow([decl("notes", "episodic")]), { supplied: { notes: mine } });
    expect(registry.get("notes")).toBe(mine);
  }
  expect(log).toEqual([]);
});

test("factories build custom types with config", async () => {
  const flow = hostFlow([decl("vec", "vector", { top_k: 3n })]);
  const registry = await buildMemories(flow, { factories: { vector: (config) => new Custom(config) } });
  expect((registry.get("vec") as Custom).config).toEqual({ top_k: 3 });
  expect(registry.types.get("vec")).toBe("vector");

  // An async factory is awaited.
  const later = await buildMemories(flow, { factories: { vector: async (config) => new Custom(config) } });
  expect(later.get("vec")).toBeInstanceOf(Custom);
});

test.each([
  ["no factory", [decl("vec", "vector")], {}, /no factory/],
  ["built-in factory", [decl("notes", "episodic")], { factories: { episodic: () => new Custom() } }, /built-in/],
  [
    "supplied with config",
    [decl("notes", "episodic", { max_episodes: 5n })],
    { supplied: { notes: new Custom() } },
    /config/,
  ],
  ["supplied undeclared", [decl("notes", "episodic")], { supplied: { other: new Custom() } }, /not declared/],
  ["storage undeclared", [decl("notes", "episodic")], { storage: { other: "x.sqlite" } }, /not declared/],
  [
    "storage for supplied",
    [decl("notes", "episodic")],
    { supplied: { notes: new Custom() }, storage: { notes: "x" } },
    /storage/,
  ],
  [
    "storage for custom",
    [decl("vec", "vector")],
    { factories: { vector: () => new Custom() }, storage: { vec: "x" } },
    /storage/,
  ],
  ["memory path", [decl("notes", "episodic")], { storage: { notes: ":memory:" } }, /not a local file path/],
  ["file uri", [decl("about", "profile")], { storage: { about: "file:about.md" } }, /not a local file path/],
  ["supplied not a memory", [decl("notes", "episodic")], { supplied: { notes: {} } }, /not a Memory/],
  ["factory not a memory", [decl("vec", "vector")], { factories: { vector: () => ({}) } }, /not a Memory/],
  [
    "factory rejects config",
    [decl("vec", "vector", { bad: 1n })],
    {
      factories: {
        vector: () => {
          throw new RangeError("unexpected bad");
        },
      },
    },
    /rejected its config/,
  ],
  ["toString type", [decl("vec", "toString")], {}, /custom type 'toString' has no factory/],
  [
    "async factory rejects config",
    [decl("vec", "vector")],
    {
      factories: {
        vector: async () => {
          throw new TypeError("missing top_k");
        },
      },
    },
    /vector rejected its config: missing top_k/,
  ],
] as const)("setup errors (%s)", async (_label, memories, options, message) => {
  await setupError(() => buildMemories(hostFlow(memories), options as never), message);
});

test("setup error messages use the TS spellings", async () => {
  const flow = hostFlow([decl("notes", "episodic")]);
  await setupError(
    () => buildMemories(flow, { supplied: { other: new Custom() } }),
    /^supplied names memory key\(s\) \["other"\] not declared by this flow \(declared: \["notes"\]\)$/,
  );
  await setupError(
    () => buildMemories(flow, { factories: { episodic: () => new Custom() } }),
    /^factories may not name built-in memory type\(s\) \["episodic"\]; use supplied to replace one built-in instance$/,
  );
  await setupError(
    () => buildMemories(flow, { supplied: { notes: new Custom() }, storage: { notes: "x" } }),
    /^memory 'notes': storage applies only to built-ins this call constructs; this key is supplied$/,
  );
  await setupError(
    () => buildMemories(flow, { supplied: { notes: 3 as never } }),
    /^memory 'notes': supplied value is not a Memory \(number\)$/,
  );
  await setupError(
    () => buildMemories(hostFlow([decl("vec", "vector")]), { factories: { vector: () => [] as never } }),
    /^memory 'vec': factory for 'vector' returned a Array, not a Memory$/,
  );
});

test.each([
  ["number", "episodic", 123],
  ["Buffer", "profile", Buffer.from("notes.md")],
  ["URL", "episodic", new URL("file:///notes.sqlite")],
  ["null", "reflective", null],
] as const)("storage values must be text paths (%s)", async (typeName, type, value) => {
  await setupError(
    () => buildMemories(hostFlow([decl("notes", type)]), { storage: { notes: value as never } }),
    new RegExp(`^storage\\["notes"\\] must be a string path, not ${typeName}$`),
  );
});

test("storage path errors show the plain path", async () => {
  const flow = hostFlow([decl("notes", "episodic")]);
  const error = await setupError(() => buildMemories(flow, { storage: { notes: "file:x" } }));
  expect(error.message).toContain("storage 'file:x' is not a local file path");
  expect(error.cause).toBeInstanceOf(MemoryOperationError);
});

test("flow must be a ComposedFlow", async () => {
  const error = await caught(() => buildMemories({} as never));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toMatch(/ComposedFlow/);
});

test("setup errors carry the declaration origin", async () => {
  const origin = new OriginRef("flow/long_term_memory/vec");
  const error = await setupError(() => buildMemories(hostFlow([decl("vec", "vector", {}, origin)])));
  expect(error.origin).toBe(origin);
  expect(error.related).toEqual([]);
});

test("a wrapped config failure keeps its cause", async () => {
  const refusal = new RangeError("unexpected bad");
  const error = await setupError(() =>
    buildMemories(hostFlow([decl("vec", "vector")]), {
      factories: {
        vector: () => {
          throw refusal;
        },
      },
    }),
  );
  expect(error.message).toBe("memory 'vec': vector rejected its config: unexpected bad");
  expect(error.cause).toBe(refusal);
});

test("flow without memory builds an empty registry", async () => {
  await using registry = await buildMemories(hostFlow([], { flowId: "none" }));
  expect(registry.size).toBe(0);
});

test("null records read as empty", async () => {
  // As Python's `None` for these options.
  const options = { storage: null, supplied: null, factories: null };
  await using nulls = await buildMemories(hostFlow([decl("notes", "episodic")]), options);
  expect(nulls.get("notes")).toBeInstanceOf(EpisodicMemory);
});

test("root layout follows flow id, scope and name", async () => {
  const root = path.join(tmp, "mem");
  {
    await using registry = await buildMemories(importFlow(), { root });
    expect([...registry.keys()]).toEqual(["about", "lib.notes", "notes"]);
    expect(fs.existsSync(root)).toBe(false); // construction is effect-free
    for (const key of ["notes", "lib.notes", "about"]) await registry.get(key)!.openRun();
    expect(fs.statSync(path.join(root, "app", "notes.sqlite")).isFile()).toBe(true);
    expect(fs.statSync(path.join(root, "app", "lib", "notes.sqlite")).isFile()).toBe(true);
  }
  expect(fs.existsSync(path.join(root, "app", ".about.md.lock"))).toBe(true);
});

test("an empty root derives paths relative to the current directory", async () => {
  const before = process.cwd();
  process.chdir(tmp);
  try {
    await using registry = await buildMemories(hostFlow([decl("notes", "episodic")]), { root: "" });
    await registry.get("notes")!.openRun();
    expect(fs.statSync(path.join(tmp, "host-flow", "notes.sqlite")).isFile()).toBe(true);
  } finally {
    process.chdir(before);
  }
});

test("root must be a string", async () => {
  const error = await caught(() => buildMemories(hostFlow([decl("notes", "episodic")]), { root: 3 as never }));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("root must be a string path, not number");
});

test("explicit storage overrides root and needs an existing parent", async () => {
  const explicit = path.join(tmp, "missing", "notes.sqlite");
  {
    const options = { root: path.join(tmp, "mem"), storage: { notes: explicit } };
    await using registry = await buildMemories(importFlow(), options);
    expect(await operationCode(() => registry.get("notes")!.openRun())).toBe("storage.io");
    expect(fs.existsSync(path.dirname(explicit))).toBe(false);
    await registry.get("lib.notes")!.openRun();
  }
  expect(fs.statSync(path.join(tmp, "mem", "app", "lib", "notes.sqlite")).isFile()).toBe(true);
});

test("two keys may not share a storage path", async () => {
  const flow = hostFlow([decl("a", "profile"), decl("b", "profile")]);
  for (const other of ["p.md", "P.md", "sub/../p.md"]) {
    await setupError(
      () => buildMemories(flow, { storage: { a: path.join(tmp, "p.md"), b: `${tmp}/${other}` } }),
      /memory 'b'.*also used by memory 'a'/,
    );
  }
  expect(entries(tmp)).toEqual([]);
});

test("explicit storage may not reuse a root-derived path", async () => {
  const root = path.join(tmp, "mem");
  const taken = path.join(root, "app", "notes.sqlite");
  await setupError(
    () => buildMemories(importFlow(), { root, storage: { "lib.notes": taken } }),
    /memory 'notes'.*also used by memory 'lib.notes'/,
  );
  expect(fs.existsSync(root)).toBe(false);
});

test("no root means RAM and no flow id is needed", async () => {
  const before = process.cwd();
  process.chdir(tmp);
  try {
    const flow = hostFlow([decl("notes", "episodic"), decl("about", "profile")], { flowId: null });
    await using registry = await buildMemories(flow);
    await registry.get("notes")!.openRun();
    await registry.get("about")!.openRun();
  } finally {
    process.chdir(before);
  }
  expect(entries(tmp)).toEqual([]);
});

test.each([[null], ["."], [".."]] as const)("root requires a safe flow id (%p)", async (flowId) => {
  await setupError(() => buildMemories(hostFlow([decl("notes", "episodic")], { flowId }), { root: tmp }), /flow id/);
});

test.each([["a/b"], ["a\\b"], ["a\0b"], [""]] as const)(
  "unsafe flow ids are rejected before touching disk (%p)",
  async (flowId) => {
    const flow = hostFlow([decl("notes", "episodic")]);
    Object.assign(flow.compiled, { flowId }); // bypass the constructor's id check
    await setupError(() => buildMemories(flow, { root: tmp }), /cannot name a directory under the memory root/);
    expect(entries(tmp)).toEqual([]);
  },
);

test("partial failure closes what was built in reverse", async () => {
  const log: string[] = [];
  const flow = hostFlow([decl("a", "alpha"), decl("b", "beta"), decl("c", "gamma")]);
  const factories = {
    alpha: () => new Closable(log, "a"),
    beta: () => new Closable(log, "b", failureWithNote("b close", "b detail")),
    gamma: () => {
      throw new RangeError("gamma refused");
    },
  };
  const error = await setupError(() => buildMemories(flow, { factories }), /gamma refused/);
  expect(log).toEqual(["b", "a"]);
  expect(error.notes).toEqual(["Memory cleanup failed: Error: b close", "b detail"]);
});

test("factory result that is not a memory is closed", async () => {
  const log: string[] = [];
  const notAMemory = { close: () => void log.push("closed") };
  await setupError(
    () => buildMemories(hostFlow([decl("vec", "vector")]), { factories: { vector: () => notAMemory as never } }),
    /not a Memory/,
  );
  expect(log).toEqual(["closed"]);
});

test("second registry on one root is busy until the first closes", async () => {
  const flow = hostFlow([decl("notes", "episodic")]);
  await using first = await buildMemories(flow, { root: tmp });
  await first.get("notes")!.openRun();
  await using second = await buildMemories(flow, { root: tmp });
  expect(await operationCode(() => second.get("notes")!.openRun())).toBe("storage.busy");
  await first.close();
  await second.get("notes")!.openRun();
});

test.each([["episodic"], ["reflective"], ["profile"]] as const)(
  "open after close is closed and opens no file (%s)",
  async (type) => {
    const registry = await buildMemories(hostFlow([decl("notes", type)]), { root: tmp });
    await registry.close();
    const memory = registry.get("notes")!;
    expect(await operationCode(() => memory.openRun())).toBe(`${type}.closed`);
    expect(await operationCode(() => memory.reset())).toBe(`${type}.closed`);
    expect(entries(tmp)).toEqual([]);
  },
);

/**
 * A loaded flow declaring `memory` around one CODE node (Python's `_flow`, loaded as Python loads
 * it, so the run has edges to follow).
 */
function runnableFlow(memory: string): Promise<ComposedFlow> {
  return loadFlow(
    `id: host-flow\nname: host\nlong_term_memory: ${memory}\nnodes:\n` +
      "  echo: {kind: code, code: return 1, output: int}\noutput: ${echo.output}\n",
  );
}

/** `runFlow` with no inputs, in an environment holding only the test's credential store. */
function run(flow: ComposedFlow, memory: unknown) {
  const env = new Map([["YIAN_AUTH_FILE", path.join(tmp, "auth.json")]]);
  return runFlow(flow, {}, { memory: memory as Map<string, Memory>, env: (name) => env.get(name) });
}

test("a plain object is refused; a Map runs", async () => {
  const flow = await runnableFlow("{notes: episodic}");
  const refused = await caught(() => run(flow, { notes: new FakeMemory() }));
  expect(refused).toBeInstanceOf(TypeError);
  expect((refused as Error).message).toBe("memory must be a Map of memory keys to backends");
  const memory = new FakeMemory();
  expect((await run(flow, new Map([["notes", memory]]))).status).toBe("succeeded");
  expect(memory.sessions.map((session) => session.closed)).toEqual([true]);
});

test("registry built for another flow is rejected", async () => {
  await using other = await buildMemories(await runnableFlow("{notes: reflective}"));
  const flow = await runnableFlow("{notes: episodic}");
  await setupError(() => run(flow, other), /'notes'.*'episodic'.*'reflective'/);
});

test("registry for this flow runs", async () => {
  const flow = await runnableFlow("{notes: episodic}");
  await using registry = await buildMemories(flow);
  expect((await run(flow, registry)).status).toBe("succeeded");
});
