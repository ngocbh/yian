/**
 * Build the memories a flow declares and own their lifetime for one host.
 *
 * This module sits beside `run.ts`, not under `memory/`, because it reads the compiled memory
 * table (`compile`) and `memory/` may import only lower memory modules. `runFlow` never constructs
 * memory; a host calls `buildMemories` and passes the returned registry as `memory`.
 */

import { type ResolvedMemory, resolveMemoryLayout } from "./compile/memory_layout.ts";
import { ComposedFlow } from "./compose/build/pipeline.ts";
import { type Env, processEnv } from "./defaults.ts";
import { addNote, compareCodeUnits, reprText } from "./diagnostics.ts";
import { ModelCallRetried, ReplayDegraded } from "./events.ts";
import type { RetryInfo } from "./llm/executor.ts";
import type { ModelAccess } from "./llm_clients/factory.ts";
import {
  BUILTIN_MEMORY_CONFIG_KEYS,
  isMemory,
  type Memory,
  MemoryOperationError,
  MemorySetupError,
} from "./memory/contracts.ts";
import { EpisodicMemory } from "./memory/episodic.ts";
import { ProfileMemory } from "./memory/profile.ts";
import { ReflectiveMemory } from "./memory/reflective.ts";
import { FileMarkdownStore, InMemoryMarkdownStore } from "./memory/storage/markdown.ts";
import { realStoragePath } from "./memory/storage/sqlite.ts";
import { jsTypeName, toHost, type Value, writeAsJson } from "./typesys/values.ts";

/** A constructor for one custom memory type, called with the declared config. */
export type MemoryFactory = (config: Record<string, unknown>) => Memory | Promise<Memory>;

/** Where the learners of one build get their environment and models and send their events. */
interface Learners {
  env: Env;
  models: ModelAccess | null;
  onEvent: ((event: ModelCallRetried | ReplayDegraded) => void) | null;
}

/** The node id a learner's model and events are attributed to. */
const LEARNER_NODE = "__memory__";

/** The `env`, `createModel` and `onRetry` options of the learner of the memory under `key`. */
function learnerOptions(key: string, { env, models, onEvent }: Learners) {
  const consumer = `memory:${key}`;
  const createModel =
    models === null
      ? null
      : (config: Readonly<Record<string, unknown>>) => {
          const degraded = new Set<string>();
          return models.createModel(config, {
            nodeId: LEARNER_NODE,
            consumer,
            // A learner's call can't be stopped.
            signal: new AbortController().signal,
            onReplayDegraded: (reason) => {
              if (degraded.has(reason)) return;
              degraded.add(reason);
              onEvent?.(new ReplayDegraded({ nodeId: LEARNER_NODE, reason }));
            },
          });
        };
  const onRetry =
    onEvent === null
      ? null
      : ({ attempt, error: { kind, message }, retryIn }: RetryInfo) =>
          onEvent(new ModelCallRetried({ nodeId: LEARNER_NODE, consumer, attempt, error: { kind, message }, retryIn }));
  return { env: models === null ? env : models.env, createModel, onRetry };
}

/**
 * @internal Attach one cleanup failure to `target` as the note `<label> <name>: <message>`, then
 * the notes the failure carries.
 */
export function noteCleanup(target: unknown, cleanup: unknown, label: string): void {
  addNote(target, () => {
    const [name, message] =
      cleanup instanceof Error ? [cleanup.name, cleanup.message] : [jsTypeName(cleanup), String(cleanup)];
    return `${label} ${name}: ${message}`;
  });
  const notes = (cleanup as { notes?: unknown } | null | undefined)?.notes;
  if (Array.isArray(notes)) for (const note of notes) addNote(target, () => String(note));
}

/**
 * Close each object that has a callable `close`, awaiting each; return the first failure.
 *
 * Later failures are attached to the first one as notes. Objects whose close threw or rejected
 * are appended to `failed` when it is given.
 */
async function closeAll(objects: Iterable<unknown>, failed?: unknown[]): Promise<{ error: unknown } | null> {
  let first: { error: unknown } | null = null;
  for (const item of objects) {
    const close = (item as { close?: unknown } | null | undefined)?.close;
    if (typeof close !== "function") continue;
    try {
      await close.call(item);
    } catch (error) {
      failed?.push(item);
      if (first === null) first = { error };
      else noteCleanup(first.error, error, "Additional memory cleanup failed:");
    }
  }
  return first;
}

/** A `Map` filled once at construction; `set`, `delete`, `clear` and the upserts throw. */
class ReadonlyTable<K, V> extends Map<K, V> {
  /** Copy `entries`, in order. */
  constructor(entries: Iterable<readonly [K, V]>) {
    // Filled with `super.set`: `Map`'s constructor would call the overridden `set`.
    super();
    for (const [key, value] of entries) super.set(key, value);
  }

  /** @throws TypeError always. */
  override set(_key: K, _value: V): this {
    throw this.readOnly();
  }

  /** @throws TypeError always. */
  override delete(_key: K): boolean {
    throw this.readOnly();
  }

  /** @throws TypeError always. */
  override clear(): void {
    throw this.readOnly();
  }

  // Bun ships `Map`'s upsert methods, which write through the internal slot, not `set`.
  /** @throws TypeError always. */
  getOrInsert(_key: K, _value: V): V {
    throw this.readOnly();
  }

  /** @throws TypeError always. */
  getOrInsertComputed(_key: K, _compute: (key: K) => V): V {
    throw this.readOnly();
  }

  private readOnly(): TypeError {
    return new TypeError("the memory registry and its types are read-only");
  }
}

/** Read-only memory-key → backend map that owns what it constructed. */
export class MemoryRegistry extends ReadonlyTable<string, Memory> {
  /**
   * Declared type of each key, from the flow's memory table. `runFlow` compares it with the table
   * of the flow it runs.
   */
  readonly types: ReadonlyMap<string, string>;
  #owned: readonly object[];
  #closing: Promise<void> | null = null;

  /**
   * Wrap built memories; `owned` lists, in build order, what `close()` closes.
   *
   * @param memories Backend for each memory key.
   * @param types Declared type for each memory key.
   * @param options.owned Objects this registry constructed. Supplied objects stay with the caller.
   */
  constructor(
    memories: Iterable<readonly [string, Memory]>,
    types: Iterable<readonly [string, string]>,
    { owned = [] }: { owned?: readonly object[] } = {},
  ) {
    super(memories);
    this.types = new ReadonlyTable(types);
    this.#owned = [...owned];
  }

  /**
   * Close owned backends in reverse build order, awaiting each.
   *
   * A backend that closed is not closed again. One whose close threw or rejected stays owned, so a
   * later call retries it; once all closed, calls do nothing. A call made while a close is in
   * flight returns that close's promise.
   *
   * @throws the first close failure, with later ones as notes.
   */
  close(): Promise<void> {
    if (this.#closing !== null) return this.#closing;
    // Set `#closing` before any close runs, so a reentrant call joins it instead of closing again.
    const { promise, resolve } = Promise.withResolvers<void>();
    this.#closing = promise;
    resolve(this.#closeOwned());
    return promise;
  }

  async #closeOwned(): Promise<void> {
    try {
      const failed: object[] = [];
      const first = await closeAll([...this.#owned].reverse(), failed);
      this.#owned = failed.reverse();
      if (first !== null) throw first.error;
    } finally {
      this.#closing = null;
    }
  }

  /** Close the registry, for `await using`. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }
}

/** Attribute one setup failure to the key's first declaration site. */
function setupError(memory: ResolvedMemory, message: string, options: { cause?: unknown } = {}): MemorySetupError {
  return new MemorySetupError(`memory ${reprText(memory.key)}: ${message}`, {
    origin: memory.origin,
    related: memory.origins.slice(1).filter((origin) => origin !== null),
    ...options,
  });
}

/**
 * Construct one built-in backend, its YAML config keys in camelCase; a reflective or profile
 * learner gets its environment, model and retry observer from `learners`.
 *
 * @throws as the backend's constructor does.
 */
function builtin(memory: ResolvedMemory, path: string | null, createParent: boolean, learners: Learners): Memory {
  const { type, config } = memory;
  const camel = (key: string) => key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
  const options = Object.fromEntries(Object.entries(config).map(([key, value]) => [camel(key), value]));
  if (type === "episodic") return new EpisodicMemory({ ...options, storage: path, createParent });
  const learner = learnerOptions(memory.key, learners);
  if (type === "reflective") return new ReflectiveMemory({ ...options, storage: path, createParent, ...learner });
  const store = path === null ? new InMemoryMarkdownStore() : new FileMarkdownStore(path, { createParent });
  return new ProfileMemory(store, { ...options, ...learner });
}

/**
 * Build one backend for every memory key the flow declares.
 *
 * For each key, in key order: `supplied[key]` wins; else a custom type is built by
 * `factories[type](config)`; else the built-in type is built from `config` with storage from
 * `storage[key]`, else `<root>/<flow id>/<scope...>/<name>.sqlite` (`.md` for profile, parent
 * created on first use), else process RAM. Nothing touches the filesystem until a backend's first
 * operation.
 *
 * @param flow Loaded flow whose memory table is read.
 * @param options.root Directory holding per-flow memory files. Requires the flow to have a safe
 *   `id`.
 * @param options.storage Explicit file per built-in key. The parent must already exist.
 * @param options.supplied Ready objects by key. The registry does not close them.
 * @param options.factories Constructors for custom types, called with the declared config as a
 *   host value (safe ints as `number`) and awaited.
 * @param options.env The environment reflective and profile memory read model defaults and
 *   credentials from; default `processEnv`. `models.env` wins when `models` is given.
 * @param options.models Builds the reflective and profile learners' models, under node
 *   `"__memory__"` with consumer `"memory:<key>"`; their calls can't be stopped. Unset, a learner
 *   builds through `chatModelFromConfig` with the environment.
 * @param options.onEvent Receives each learner retry as a `ModelCallRetried` (the error scrubbed
 *   of the call's credentials) and, with `models`, each reason a learner's model gives for leaving
 *   out replay state as one `ReplayDegraded` per call, both under node `"__memory__"`. Learners run
 *   outside any run, so their events reach this callback, never a run's `onEvent`. A throw from it
 *   during a retry fails that learner call (`reflective.provider` or `profile.provider`).
 * @returns Map from memory key to backend, plus declared types.
 * @throws TypeError when `flow` isn't a `ComposedFlow` or `root` isn't a string.
 * @throws MemorySetupError when the table is invalid, or a key can't be built as asked (including
 *   a storage value that isn't a string, a path that isn't a local file path, or two built-in keys
 *   whose storage paths are equal, ignoring case). Anything already built is closed before the
 *   error propagates.
 */
export async function buildMemories(
  flow: ComposedFlow,
  {
    root = null,
    storage = {},
    supplied = {},
    factories = {},
    env = processEnv,
    models = null,
    onEvent = null,
  }: {
    root?: string | null;
    storage?: Readonly<Record<string, string>> | null;
    supplied?: Readonly<Record<string, Memory>> | null;
    factories?: Readonly<Record<string, MemoryFactory>> | null;
    env?: Env;
    models?: ModelAccess | null;
    onEvent?: ((event: ModelCallRetried | ReplayDegraded) => void) | null;
  } = {},
): Promise<MemoryRegistry> {
  if (!(flow instanceof ComposedFlow)) throw new TypeError("flow must be a ComposedFlow");
  let base: string | null = null;
  if (root !== null) {
    if (typeof root !== "string") throw new TypeError(`root must be a string path, not ${jsTypeName(root)}`);
    const flowId = flow.compiled.flowId;
    if (flowId === null) {
      throw new MemorySetupError(
        "root needs the flow to have an id: field; its memory lives under <root>/<flow id>/",
      );
    }
    if (flowId === "" || flowId === "." || flowId === ".." || /[/\\\0]/.test(flowId)) {
      throw new MemorySetupError(`flow id ${reprText(flowId)} cannot name a directory under the memory root`);
    }
    // Joined as text, never `path.join`, which folds `..` without looking at symlinks. An empty
    // root adds no segment and a trailing `/` is dropped, as Python's `Path` reads them.
    base = root === "" ? flowId : [root.replace(/\/+$/, ""), flowId].join("/");
  }
  // Copied into maps so a key such as `toString` never finds `Object.prototype`.
  const paths = new Map<string, unknown>(Object.entries(storage ?? {}));
  const given = new Map<string, unknown>(Object.entries(supplied ?? {}));
  const makers = new Map<string, unknown>(Object.entries(factories ?? {}));
  const table = resolveMemoryLayout(flow.compiled).memories;
  const declared = new Set(table.map((memory) => memory.key));
  for (const [name, keys] of [
    ["storage", paths],
    ["supplied", given],
  ] as const) {
    const extra = [...keys.keys()].filter((key) => !declared.has(key)).sort(compareCodeUnits);
    if (extra.length > 0) {
      throw new MemorySetupError(
        `${name} names memory key(s) ${writeAsJson(extra)} not declared by this flow ` +
          `(declared: ${writeAsJson([...declared].sort(compareCodeUnits))})`,
      );
    }
  }
  for (const [key, path] of paths) {
    // Backends take text paths only. Checked here so other values are not reported as a config
    // error, and null does not silently mean RAM.
    if (typeof path !== "string") {
      throw new MemorySetupError(`storage[${writeAsJson(key)}] must be a string path, not ${path === null ? "null" : jsTypeName(path)}`);
    }
  }
  const shadowed = [...makers.keys()].filter((type) => BUILTIN_MEMORY_CONFIG_KEYS.has(type)).sort(compareCodeUnits);
  if (shadowed.length > 0) {
    throw new MemorySetupError(
      `factories may not name built-in memory type(s) ${writeAsJson(shadowed)}; ` +
        "use supplied to replace one built-in instance",
    );
  }
  const memories = new Map<string, Memory>();
  const owned: object[] = [];
  const seen = new Map<string, string>();
  try {
    for (const memory of table) {
      memories.set(
        memory.key,
        await buildOne(memory, paths as Map<string, string>, given, makers, owned, base, seen, { env, models, onEvent }),
      );
    }
  } catch (primary) {
    // Construction is effect-free, so this cleanup does no I/O.
    const cleanup = await closeAll([...owned].reverse());
    if (cleanup !== null) noteCleanup(primary, cleanup.error, "Memory cleanup failed:");
    throw primary;
  }
  return new MemoryRegistry(
    memories,
    table.map((memory) => [memory.key, memory.type] as const),
    { owned },
  );
}

/**
 * Return the backend for one key, appending it to `owned` if built here.
 *
 * `seen` maps each normalized built-in storage path seen so far to its key.
 */
async function buildOne(
  memory: ResolvedMemory,
  storage: ReadonlyMap<string, string>,
  supplied: ReadonlyMap<string, unknown>,
  factories: ReadonlyMap<string, unknown>,
  owned: object[],
  base: string | null,
  seen: Map<string, string>,
  learners: Learners,
): Promise<Memory> {
  if (supplied.has(memory.key)) {
    if (storage.has(memory.key)) {
      throw setupError(memory, "storage applies only to built-ins this call constructs; this key is supplied");
    }
    if (Object.keys(memory.config).length > 0) {
      throw setupError(memory, "a supplied memory cannot take YAML config; remove config or build it with factories");
    }
    const value = supplied.get(memory.key);
    if (!isMemory(value)) throw setupError(memory, `supplied value is not a Memory (${jsTypeName(value)})`);
    return value;
  }
  if (!BUILTIN_MEMORY_CONFIG_KEYS.has(memory.type)) {
    if (storage.has(memory.key)) {
      throw setupError(memory, `storage applies only to built-in types, not ${reprText(memory.type)}`);
    }
    const factory = factories.get(memory.type);
    if (factory === undefined || factory === null) {
      throw setupError(memory, `custom type ${reprText(memory.type)} has no factory or supplied value`);
    }
    let value: unknown;
    try {
      value = await (factory as MemoryFactory)(toHost(memory.config as Value) as Record<string, unknown>);
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) {
        throw setupError(memory, `${memory.type} rejected its config: ${error.message}`, { cause: error });
      }
      throw error;
    }
    if (!isMemory(value)) {
      const error = setupError(
        memory,
        `factory for ${reprText(memory.type)} returned a ${jsTypeName(value)}, not a Memory`,
      );
      // The factory built it, so this call owns it even though it is rejected.
      const cleanup = await closeAll([value]);
      if (cleanup !== null) noteCleanup(error, cleanup.error, "Memory cleanup failed:");
      throw error;
    }
    owned.push(value);
    return value;
  }
  let path = storage.get(memory.key) ?? null;
  let createParent = false;
  if (path === null && base !== null) {
    const scope = memory.key.split(".");
    const name = scope.pop()!;
    const suffix = memory.type === "profile" ? ".md" : ".sqlite";
    path = [base, ...scope, name + suffix].join("/");
    createParent = true;
  }
  let value: Memory;
  try {
    value = builtin(memory, path, createParent, learners);
  } catch (error) {
    if (error instanceof MemoryOperationError) {
      throw setupError(memory, `storage ${reprText(path ?? "")} is not a local file path`, { cause: error });
    }
    if (error instanceof TypeError || error instanceof RangeError) {
      throw setupError(memory, `${memory.type} rejected its config: ${error.message}`, { cause: error });
    }
    throw error;
  }
  owned.push(value);
  if (path !== null) {
    // Lowercased so two keys can't share one file on a case-insensitive filesystem (JS has no
    // casefold); `realStoragePath` reads links but opens nothing.
    const normal = realStoragePath(path).toLowerCase();
    const other = seen.get(normal);
    // memoryTable lists each key once, so a path already seen belongs to another key.
    if (other !== undefined) {
      throw setupError(memory, `storage ${reprText(path)} is also used by memory ${reprText(other)}`);
    }
    seen.set(normal, memory.key);
  }
  return value;
}
