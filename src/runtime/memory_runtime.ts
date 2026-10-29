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

import { addNote, compareCodeUnits, errorTypeName, OriginRef, reprText } from "../diagnostics.ts";
import {
  isMemory,
  type JsonObject,
  LearnContext,
  type Memory,
  MemoryContent,
  MemoryContext,
  type MemoryMutation,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  type MemorySession,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  RecallContext,
  sameMemorySnapshotEnvelope,
} from "../memory/contracts.ts";
import { isExactly, sameItems } from "../nodes/contracts.ts";
import { isList } from "../typesys/values.ts";
import { MemoryProposal } from "./contracts.ts";
import { MemoryCommitRequestedRecord } from "./log.ts";

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
  if (!isList(memories)) throw new TypeError("memories must be an array of MemorySpec values");
  // Spread, so a hole of a sparse array reads as `undefined` and is refused.
  const copy = [...memories];
  if (!copy.every((spec) => isExactly(spec, MemorySpec))) {
    throw new TypeError("memories must be an array of MemorySpec values");
  }
  const keys = copy.map((spec) => spec.key);
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

/** @internal A known per-key proposal failure before any commit request. */
export class LearningProposalError extends Error {
  override name = "LearningProposalError";
  /** The memory key whose proposal failed. */
  readonly memory: string;
  /** The sanitized failure; its message is this error's message. */
  readonly failure: MemoryOperationFailure;

  constructor(memory: string, failure: MemoryOperationFailure, options?: ErrorOptions) {
    super(failure.message, options);
    this.memory = memory;
    this.failure = failure;
  }
}

/** Copy one typed backend failure, or sanitize an error whose fields were changed to invalid ones. */
function operationFailure(operation: string, error: MemoryOperationError): MemoryOperationFailure {
  try {
    return new MemoryOperationFailure(error.stage, error.code, error.message, error.retryable);
  } catch (failure) {
    if (!(failure instanceof TypeError || failure instanceof RangeError)) throw failure;
    return unexpectedOperationFailure(operation);
  }
}

/** Return one stable final failure without leaking an unexpected error. */
function unexpectedOperationFailure(operation: string): MemoryOperationFailure {
  return new MemoryOperationFailure(
    operation,
    "memory.unexpected",
    `memory ${operation} raised an unexpected error`,
    false,
  );
}

/** Return one stable final failure for a malformed memory return value. */
function invalidResultFailure(operation: string): MemoryOperationFailure {
  return new MemoryOperationFailure(
    operation,
    "memory.invalid_result",
    `memory ${operation} returned an invalid result`,
    false,
  );
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
        return `memory ${reprText(key)} session close also raised ${errorTypeName(error)}: ${message}`;
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
   * resolves; it never rejects and adds no notes.
   *
   * @throws the first session close failure, after every session was closed; later failures are
   *   noted on it.
   */
  close(): Promise<void> {
    return this.closePreserving(null);
  }

  /**
   * @internal Close keeping `primary.error` as the initiating failure:
   * every close failure is noted on it instead of thrown. `null` is `close()`.
   * A later call waits for the first close, never rejects and adds no notes; a session's `close`
   * must not await it.
   */
  closePreserving(primary: { error: unknown } | null): Promise<void> {
    if (this.#closing !== null) return this.#closing.then(() => undefined, () => undefined);
    // Set `#closing` before any session's close runs, so a reentrant call joins it instead of closing again.
    const { promise, resolve } = Promise.withResolvers<void>();
    this.#closing = promise;
    resolve(closeSessions(this.sessions, primary));
    return promise;
  }

  /**
   * Recall each key in the scope into ordered sections.
   *
   * Sections keep the scope's outermost-first key order; a key with empty notes contributes none.
   *
   * @param scope - The exact compiled consumer scope over opened keys.
   * @param recall - The matching recall context.
   * @param options.signal - Checked after recall.
   * @throws Error when the runtime is closed or the scope names a key that wasn't opened.
   * @throws TypeError when a carrier or the signal has the wrong type.
   * @throws RangeError when the paths disagree.
   * @throws MemoryRecallError when a memory rejects recall or returns invalid content.
   * @throws `signal.reason` when the signal aborted by the end of recall; any other recall failure
   *   propagates unchanged.
   */
  async openContext(
    scope: MemoryScope,
    recall: RecallContext,
    options: { signal: AbortSignal },
  ): Promise<MemoryContext> {
    if (this.closed) throw new Error("memory runtime is closed");
    if (!isExactly(scope, MemoryScope)) throw new TypeError("memory context scope must be a MemoryScope");
    if (!isExactly(recall, RecallContext)) throw new TypeError("memory recall must be a RecallContext");
    const signal = options?.signal;
    if (!(signal instanceof AbortSignal)) throw new TypeError("signal must be an AbortSignal");
    if (!sameItems(recall.consumerStaticPath, scope.staticPath)) {
      throw new RangeError("memory recall context does not match the consumer scope");
    }
    if (!scope.memories.every((key) => this.sessions.has(key))) {
      throw new Error("memory scope names a memory outside the opened table");
    }
    const sections: MemorySection[] = [];
    for (const key of scope.memories) {
      const origin = this.origins.get(key)!;
      let recalled: unknown;
      try {
        recalled = await this.sessions.get(key)!.recall(recall);
      } catch (error) {
        if (!(error instanceof MemoryOperationError)) throw error;
        throw new MemoryRecallError(`memory ${reprText(key)} recall failed: ${error.message}`, {
          origin,
          cause: error,
        });
      }
      let section: MemorySection;
      try {
        let text: string;
        if (typeof recalled === "string") text = recalled;
        else if (isExactly(recalled, MemoryContent)) text = recalled.text;
        else throw new TypeError("memory recall must return a string or a MemoryContent");
        section = new MemorySection(key, text);
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        // The content may be private, so the message leaves it out.
        throw new MemoryRecallError(`memory ${reprText(key)} returned invalid recall content`, {
          origin,
          cause: error,
        });
      }
      if (section.text) sections.push(section);
    }
    signal.throwIfAborted();
    return new MemoryContext({ sections });
  }

  /**
   * Stage every key's proposal without writing records or memory.
   *
   * Each learner receives the very context passed in; a context is frozen, so it needs no copy.
   *
   * @param contexts - One context per recorded key, in table order, including empty batches.
   * @returns The validated mutations in table order, as a frozen array.
   * @throws Error when the runtime is closed.
   * @throws TypeError when `contexts` isn't an array of exact `LearnContext` values.
   * @throws RangeError when the contexts don't cover the table in order or belong to several runs.
   * @throws LearningProposalError when a learner throws a `MemoryOperationError`, whatever its
   *   `cause`, or returns an invalid proposal; any other learner error propagates unchanged.
   */
  async proposeLearning(contexts: readonly LearnContext[]): Promise<readonly MemoryProposal[]> {
    if (this.closed) throw new Error("memory runtime is closed");
    const message = "learning contexts must be an array of exact LearnContext values";
    if (!isList(contexts)) throw new TypeError(message);
    // Spread, so a hole of a sparse array reads as `undefined` and is refused, and later edits to
    // the caller's array across the awaits below change nothing.
    const copy = [...contexts];
    if (!copy.every((context) => isExactly(context, LearnContext))) throw new TypeError(message);
    const keys = this.memories.map((spec) => spec.key);
    if (!sameItems(copy.map((context) => context.memory), keys)) {
      throw new RangeError("learning contexts must cover the complete recorded memory table in order");
    }
    if (new Set(copy.map((context) => context.runId)).size > 1) {
      throw new RangeError("learning contexts must belong to one run");
    }

    const proposals: MemoryProposal[] = [];
    for (const context of copy) {
      const key = context.memory;
      let raw: unknown;
      try {
        raw = await this.sessions.get(key)!.learn(context);
      } catch (error) {
        if (!(error instanceof MemoryOperationError)) throw error;
        throw new LearningProposalError(key, operationFailure("learn", error), { cause: error });
      }
      try {
        if (!isList(raw)) throw new TypeError("memory learn() must return an array");
        proposals.push(new MemoryProposal(key, raw as readonly MemoryMutation[]));
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw new LearningProposalError(key, invalidResultFailure("learn"), { cause: error });
      }
    }
    return Object.freeze(proposals);
  }

  /**
   * Perform one already accepted backward request without writing its observation.
   *
   * @param request - The accepted attempt-aware request, with its memory key, stable id and
   *   ordered batch.
   * @returns A known backend or return-contract failure, or `null` on success.
   * @throws Error when the runtime is closed.
   * @throws TypeError when `request` isn't exactly a `MemoryCommitRequestedRecord`.
   * @throws RangeError when the request names a key that wasn't opened.
   * @throws any commit error other than a `MemoryOperationError`, unchanged.
   */
  async commitLearning(request: MemoryCommitRequestedRecord): Promise<MemoryOperationFailure | null> {
    if (this.closed) throw new Error("memory runtime is closed");
    if (!isExactly(request, MemoryCommitRequestedRecord)) {
      throw new TypeError("backward commit requires an exact MemoryCommitRequestedRecord");
    }
    if (!this.sessions.has(request.memory)) {
      throw new RangeError("backward commit requires an enabled memory and attempt");
    }
    let result: unknown;
    try {
      result = await this.sessions.get(request.memory)!.commit(request.commitId, request.mutations);
    } catch (error) {
      if (!(error instanceof MemoryOperationError)) throw error;
      return operationFailure("commit", error);
    }
    return result === undefined || result === null ? null : invalidResultFailure("commit");
  }

  /**
   * Validate and open one session per recorded memory key.
   *
   * `snapshots` `null` or omitted opens the implementations at their current state for a fresh
   * run. An array restores the exact public snapshots recorded by `RunStarted` during recovery. An
   * empty table opens nothing. `signal`, the run's stop, reaches each backend's `openRun`.
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
    signal?: AbortSignal | null;
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
    const signal = options.signal ?? null;

    const opened = new Map<string, MemorySession>();
    const captured: MemorySnapshotEnvelope[] = [];
    try {
      for (const key of keys) {
        const origin = origins.get(key)!;
        try {
          const snapshot = restore.get(key) ?? null;
          const options = signal === null ? { snapshot } : { snapshot, signal };
          const session: unknown = await registry.get(key)!.openRun(options);
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

    return new this({ memories, origins, sessions: opened, snapshots: Object.freeze(captured) });
  }
}
