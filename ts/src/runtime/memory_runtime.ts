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
  MemoryOperationError,
  type MemorySession,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  sameMemorySnapshotEnvelope,
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

/** Require the narrow callable session surface without reading its snapshot. */
function validateSession(value: unknown, key: string, origin: OriginRef | null): MemorySession {
  for (const method of ["recall", "learn", "commit", "close"] as const) {
    if (typeof (value as Partial<MemorySession> | null | undefined)?.[method] !== "function") {
      throw new MemorySetupError(`memory ${reprText(key)} returned an invalid run session`, { origin });
    }
  }
  return value as MemorySession;
}

// Local copy of memory/session.ts's addNote.
/**
 * Append the note `text()` to `error`'s own `notes` array, creating it when absent (Python's
 * `add_note`). A failure, such as a frozen error, a non-array `notes`, a primitive or a throwing
 * `text`, loses the note, as Python swallows a failing `add_note`.
 */
function addNote(error: unknown, text: () => string): void {
  try {
    if (typeof error !== "object" || error === null) return;
    const target = error as { notes?: unknown };
    const note = text();
    if (!Object.hasOwn(target, "notes")) target.notes = [];
    if (Array.isArray(target.notes)) target.notes.push(note);
  } catch {}
}

// Local copy of memory/session.ts's typeName.
/** Python's `type(value).__name__`: the constructor's name, else a string `name`, else `Error`. */
function typeName(value: unknown): string {
  const constructorName = (value as { constructor?: { name?: unknown } } | null | undefined)?.constructor?.name;
  if (typeof constructorName === "string" && constructorName) return constructorName;
  const name = (value as { name?: unknown } | null | undefined)?.name;
  return typeof name === "string" ? name : "Error";
}

/**
 * Close `sessions` in reverse key order, each once, while preserving an initiating failure.
 *
 * With `primary`, every close failure is noted on `primary.error` and the promise resolves.
 * Without one, the first failure is kept, later ones are noted on it, and it is thrown after every
 * session was closed.
 */
async function closeSessions(
  sessions: ReadonlyMap<string, MemorySession>,
  primary: { error: unknown } | null,
): Promise<void> {
  let first: { error: unknown } | null = null;
  for (const key of [...sessions.keys()].reverse()) {
    try {
      await sessions.get(key)!.close();
    } catch (error) {
      const target = primary ?? first;
      if (target === null) {
        first = { error };
        continue;
      }
      addNote(target.error, () => {
        const message = error instanceof Error ? error.message : String(error);
        return `memory ${reprText(key)} session close also raised ${typeName(error)}: ${message}`;
      });
    }
  }
  if (primary === null && first !== null) throw first.error;
}

/** One opened set of per-key sessions shared by execution and learning. */
export class MemoryRuntime {
  /** The frozen recorded memory table for this run. */
  readonly memories: readonly MemorySpec[];
  /** The frozen public snapshots captured after opening. */
  readonly snapshots: readonly MemorySnapshotEnvelope[];
  private readonly origins: ReadonlyMap<string, OriginRef | null>;
  private readonly sessions: ReadonlyMap<string, MemorySession>;
  #closing: Promise<void> | null = null;

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

  /** @internal Whether `close()` has started. */
  get closed(): boolean {
    return this.#closing !== null;
  }

  /**
   * Close every opened session in reverse key order exactly once.
   *
   * A later call, even while the first is still closing, waits for the first to finish and then
   * resolves; it never rejects and adds no notes (Python's returns at once).
   *
   * @throws the first session close failure, after every session was closed; later failures are
   *   noted on it.
   */
  close(): Promise<void> {
    return this.closePreserving(null);
  }

  /**
   * @internal Close keeping `primary.error` as the initiating failure (Python's
   * `_close_preserving`): every close failure is noted on it instead of thrown. `null` is `close()`.
   * A later call waits for the first close, never rejects and adds no notes.
   */
  closePreserving(primary: { error: unknown } | null): Promise<void> {
    if (this.#closing !== null) return this.#closing.then(() => undefined, () => undefined);
    // Mark the runtime closing before any session's close runs, so a reentrant close is a no-op.
    const { promise, resolve } = Promise.withResolvers<void>();
    this.#closing = promise;
    resolve(closeSessions(this.sessions, primary));
    return promise;
  }

  /**
   * Validate and open one session per recorded memory key.
   *
   * `snapshots` `null` or omitted opens the implementations at their current state for a fresh
   * run. An array restores the exact public snapshots recorded by `RunStarted` during recovery. An
   * empty table opens nothing.
   *
   * @throws TypeError or RangeError for a malformed table, origins, registry or snapshots.
   * @throws MemorySetupError when the host supplied no backend for a key, a backend fails to open
   *   or returns an invalid session, a session fails to export or exports an invalid snapshot, or a
   *   restored session does not reproduce its recorded snapshot; every session opened so far is
   *   closed first.
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
    // Copy the caller's arrays before the first await, so later edits to them change nothing.
    const memories = Object.freeze([...options.memories]);
    const recorded = options.snapshots == null ? null : [...options.snapshots];

    const opened = new Map<string, MemorySession>();
    const captured: MemorySnapshotEnvelope[] = [];
    try {
      for (const key of keys) {
        const origin = origins.get(key)!;
        try {
          const session: unknown = await registry.get(key)!.openRun({ snapshot: restore.get(key) ?? null });
          // Register anything closable first, so a session failing the surface check is still closed.
          if (typeof (session as { close?: unknown } | null | undefined)?.close === "function") {
            opened.set(key, session as MemorySession);
          }
          opened.set(key, validateSession(session, key, origin));
        } catch (error) {
          if (!(error instanceof MemoryOperationError)) throw error;
          throw new MemorySetupError(`memory ${reprText(key)} could not open its run session: ${error.message}`, {
            origin,
            cause: error,
          });
        }
      }

      for (const key of keys) {
        const origin = origins.get(key)!;
        try {
          captured.push(new MemorySnapshotEnvelope(key, opened.get(key)!.snapshot));
        } catch (error) {
          if (error instanceof MemoryOperationError) {
            throw new MemorySetupError(`memory ${reprText(key)} could not export its run snapshot: ${error.message}`, {
              origin,
              cause: error,
            });
          }
          if (error instanceof TypeError || error instanceof RangeError) {
            throw new MemorySetupError(`memory ${reprText(key)} exported an invalid run snapshot`, {
              origin,
              cause: error,
            });
          }
          throw error;
        }
      }
      if (recorded !== null) {
        const index = recorded.findIndex((envelope, i) => !sameMemorySnapshotEnvelope(envelope, captured[i]!));
        if (index >= 0) {
          throw new MemorySetupError("restored memory session did not reproduce its recorded snapshot", {
            origin: origins.get(keys[index]!)!,
          });
        }
      }
    } catch (error) {
      await closeSessions(opened, { error });
      throw error;
    }

    return new MemoryRuntime({ memories, origins, sessions: opened, snapshots: Object.freeze(captured) });
  }
}
