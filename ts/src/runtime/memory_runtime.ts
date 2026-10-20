/**
 * Run-scoped orchestration for host-injected memory implementations.
 *
 * The compile layer supplies the recorded memory table, one `MemorySpec` per key, and each key's
 * optional authored origin. This module validates the host registry, opens exactly one session per
 * key, captures the public recovery snapshots, and admits ordered memory context for the
 * evaluator. It does not resolve YAML, inspect node kinds, or construct a concrete memory backend.
 * The explicit backward path stages one proposal per key and performs accepted commit requests
 * separately; the attempt driver owns their records and ordering.
 *
 * Never imports: compile, compose, the engine, run or the CLI.
 */

import { compareCodeUnits, OriginRef, reprText } from "../diagnostics.ts";
import {
  isMemory,
  type JsonObject,
  type Memory,
  type MemorySession,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
} from "../memory/contracts.ts";
import { isExactly, sameItems } from "../nodes/contracts.ts";
import { isList } from "../typesys/values.ts";

/** Return one exact non-empty well-formed registry identity. */
function requireName(value: unknown, name: string): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (value === "") throw new RangeError(`${name} must be non-empty`);
  if (!value.isWellFormed()) throw new RangeError(`${name} must be valid UTF-8`);
  return value;
}

/** Replay the source-neutral origin contract after possible mutation. */
function revalidateOrigin(value: unknown): OriginRef | null {
  if (value === null) return null;
  if (!(value instanceof OriginRef)) throw new TypeError("memory origins must be OriginRef values or null");
  new OriginRef(value.key);
  return value;
}

/** Require exact specs in sorted unique key order; return the keys. */
function validateMemories(memories: unknown): readonly string[] {
  if (!isList(memories) || !memories.every((spec) => isExactly(spec, MemorySpec))) {
    throw new TypeError("memories must be an array of MemorySpec values");
  }
  const keys = (memories as MemorySpec[]).map((spec) => spec.key);
  if (keys.some((key, i) => i > 0 && compareCodeUnits(keys[i - 1]!, key) >= 0)) {
    throw new RangeError("memories must be sorted by key and unique");
  }
  return keys;
}

/** Return a detached origin map with one entry per memory key, in key order. */
function validateOrigins(keys: readonly string[], origins: unknown): Map<string, OriginRef | null> {
  if (!(origins instanceof Map)) throw new TypeError("memory origins must be a Map");
  for (const key of origins.keys()) {
    if (typeof key !== "string") throw new TypeError("memory origin keys must be strings");
  }
  if (origins.size !== keys.length || !keys.every((key) => origins.has(key))) {
    throw new RangeError("memory origins must exactly match the memory keys");
  }
  return new Map(keys.map((key) => [key, revalidateOrigin(origins.get(key))]));
}

/** Return a detached registry and reject one object registered under several names. */
function validateRegistry(memory: unknown): Map<string, Memory> {
  if (!(memory instanceof Map)) throw new TypeError("memory must be a Map of memory keys to backends");
  const result = new Map<string, Memory>();
  const identities = new Map<object, string>();
  for (const [rawName, implementation] of memory) {
    const name = requireName(rawName, "memory name");
    if (!isMemory(implementation)) {
      throw new TypeError(`memory ${reprText(name)} must implement the Memory protocol`);
    }
    const prior = identities.get(implementation);
    if (prior !== undefined) {
      throw new RangeError(
        `one memory object cannot be registered under multiple names: ${reprText(prior)} and ${reprText(name)}`,
      );
    }
    identities.set(implementation, name);
    result.set(name, implementation);
  }
  return result;
}

/** Return the per-key restore payload, or fresh `null` payloads. */
function validateSnapshots(snapshots: unknown, keys: readonly string[]): Map<string, JsonObject | null> {
  if (snapshots === null || snapshots === undefined) return new Map(keys.map((key) => [key, null]));
  if (!isList(snapshots)) throw new TypeError("memory snapshots must be an array or null");
  const restored = new Map<string, JsonObject | null>();
  for (const envelope of snapshots) {
    if (!isExactly(envelope, MemorySnapshotEnvelope)) {
      throw new TypeError("memory snapshots must contain MemorySnapshotEnvelope values");
    }
    if (restored.has(envelope.memory)) throw new RangeError("memory snapshot keys must be unique");
    restored.set(envelope.memory, envelope.snapshot);
  }
  if (!sameItems([...restored.keys()], keys)) {
    throw new RangeError("memory snapshots must exactly match the memory keys in canonical order");
  }
  return restored;
}

/** One opened set of per-key sessions shared by execution and learning. */
export class MemoryRuntime {
  /** The frozen recorded memory table for this run. */
  readonly memories: readonly MemorySpec[];
  /** The frozen public snapshots captured after opening. */
  readonly snapshots: readonly MemorySnapshotEnvelope[];
  private readonly origins: ReadonlyMap<string, OriginRef | null>;
  private readonly sessions: ReadonlyMap<string, MemorySession>;

  /** @internal Use `MemoryRuntime.open`, which validates and opens the sessions. */
  constructor(options: {
    memories: readonly MemorySpec[];
    origins: ReadonlyMap<string, OriginRef | null>;
    sessions: ReadonlyMap<string, MemorySession>;
    snapshots: readonly MemorySnapshotEnvelope[];
  }) {
    this.memories = options.memories;
    this.origins = options.origins;
    this.sessions = options.sessions;
    this.snapshots = options.snapshots;
  }

  /**
   * Validate and open one session per recorded memory key.
   *
   * `snapshots` `null` or omitted opens the implementations at their current state for a fresh
   * run. An array restores the exact public snapshots recorded by `RunStarted` during recovery. An
   * empty table opens nothing.
   *
   * @throws TypeError or RangeError for a malformed table, origins, registry or snapshots.
   * @throws MemorySetupError when the host supplied no backend for a key.
   */
  static async open(options: {
    memories: readonly MemorySpec[];
    origins: ReadonlyMap<string, OriginRef | null>;
    memory: ReadonlyMap<string, Memory>;
    snapshots?: readonly MemorySnapshotEnvelope[] | null;
  }): Promise<MemoryRuntime> {
    const keys = validateMemories(options.memories);
    const origins = validateOrigins(keys, options.origins);
    const registry = validateRegistry(options.memory);
    const restore = validateSnapshots(options.snapshots, keys);
    for (const key of keys) {
      if (!registry.has(key)) {
        throw new MemorySetupError(`memory ${reprText(key)} was not supplied by the host`, {
          origin: origins.get(key)!,
        });
      }
    }
    throw new Error("not implemented");
  }
}
