/**
 * Markdown profile memory: one versioned document per memory instance.
 *
 * `ProfileMemory` recalls one markdown document into every AGENT whose stack holds the instance
 * and, on backward, asks one model call to rewrite the whole document from that run's feedback.
 * Commits are compare-and-set writes to a `MarkdownStore` and never call a model.
 *
 * Knows about: `memory/contracts`, `memory/storage/markdown`, `memory/storage/sqlite`, `defaults`,
 * `typesys/json`, `typesys/values`, `llm`, `llm_clients`.
 * Never imports: the `memory` facade, nodes, runtime.
 */

import { createHash } from "node:crypto";

import { current, type Env, processEnv } from "../defaults.ts";
import type { ChatModel } from "../llm/events.ts";
import type { RetryInfo } from "../llm/executor.ts";
import { type HostConnection, popHostCredentials } from "../llm_clients/config.ts";
import { chatModelFromConfig } from "../llm_clients/factory.ts";
import type { JsonObject as StoreObject } from "../typesys/json.ts";
import { fromHost, isList, isRecord } from "../typesys/values.ts";
import {
  type JsonObject,
  type LearnContext,
  type Memory,
  MemoryMutation,
  MemoryOperationError,
  type MemorySession,
  RecallContext,
  sameMemoryMutation,
} from "./contracts.ts";
import { type MarkdownStore, MarkdownVersionMismatch } from "./storage/markdown.ts";
import { isSystemError } from "./storage/sqlite.ts";

const SNAPSHOT_FORMAT = "yian.profile_memory";
const SNAPSHOT_VERSION = 1n;
const SNAPSHOT_KEYS = ["format", "version", "profile_version", "text_sha256"] as const;
const OPEN_TAG = "<profile>";
const CLOSE_TAG = "</profile>";
const SYSTEM =
  "You maintain one markdown profile document. The user message is JSON with the " +
  "current document, the older document the reviewed run saw when it differs, " +
  "the feedback on that run, and recent agent episodes with their visible model " +
  "and tool calls. Rewrite the complete document so later runs follow the " +
  "feedback. Keep content the feedback does not contradict, including edits " +
  "people made directly. The current document may already reflect this feedback, " +
  "for example after an interrupted earlier attempt or a person's edit: never " +
  "repeat what it already says, and return it unchanged if nothing needs to " +
  "change. Stay under max_chars characters. Never call tools. Reply with the " +
  `complete new document between ${OPEN_TAG} and ${CLOSE_TAG} and nothing else.`;

/** Build the learner's model from the config Python passes to `model_from_config`. */
type CreateModel = (config: Readonly<Record<string, unknown>>) => ChatModel | Promise<ChatModel>;

/** Observe one retried learner model call. */
type OnRetry = (retry: RetryInfo) => void | Promise<void>;

/** Return the hex SHA-256 of one UTF-8 string. */
function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** Build one typed profile failure, with `cause` (Python's `raise ... from`) when given. */
function error(
  stage: string,
  code: string,
  message: string,
  options: { retryable?: boolean; cause?: unknown } = {},
): MemoryOperationError {
  const failure = new MemoryOperationError(stage, code, message, options.retryable ?? false);
  if ("cause" in options) failure.cause = options.cause;
  return failure;
}

/**
 * Map a store read failure: I/O is retryable, an unreadable document is not.
 *
 * @throws unknown `cause` unchanged when it is neither a system error nor a `RangeError`.
 */
function readError(stage: string, cause: unknown): MemoryOperationError {
  if (isSystemError(cause)) {
    return error(stage, "profile.store_unavailable", "profile store could not be read", { retryable: true, cause });
  }
  if (cause instanceof RangeError) {
    return error(stage, "profile.store_invalid", "profile store returned an unreadable document", { cause });
  }
  throw cause;
}

// Local copy of memory/contracts.ts's isExactly, which that module keeps private.
/** Whether `value` is an instance of exactly `cls`, not of a subclass. */
function isExactly<T>(value: unknown, cls: abstract new (...args: never[]) => T): value is T {
  return typeof value === "object" && value !== null && Object.getPrototypeOf(value) === cls.prototype;
}

/**
 * Return a budget as a `number`.
 *
 * @throws RangeError when it isn't a positive safe-integer `number` or `bigint`.
 */
function budget(value: unknown, name: string): number {
  const integer = typeof value === "bigint" || (typeof value === "number" && Number.isSafeInteger(value));
  if (!integer || value <= 0 || value > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`${name} must be a positive integer`);
  }
  return Number(value);
}

/**
 * Memory backend that keeps one markdown profile in a versioned store.
 *
 * Inject one instance under the memory key the flow declares, for example
 * `runFlow(flow, inputs, { memory: { profile: new ProfileMemory(store) } })`. Every AGENT whose
 * stack holds this key sees the whole document once.
 */
export class ProfileMemory implements Memory {
  /** Largest document the learner may write; host writes are not bounded. */
  readonly maxChars: number;
  /** Character budget for episode JSON sent to the learner. */
  readonly inputBudgetChars: number;
  // TS-private, not `#`, so sessions can read them, as Python's read `_store`.
  private readonly store: MarkdownStore;
  private readonly llmConfig: Record<string, unknown>;
  private readonly host: HostConnection;
  private readonly env: Env;
  private readonly createModel: CreateModel;
  private readonly onRetry: OnRetry | undefined;
  private closed = false;

  /**
   * Bind the backend to one store and learner model. An option that is `undefined` or `null` is
   * unset.
   *
   * @param store Versioned document store, such as `FileMarkdownStore`.
   * @param options.llmConfig Learner model config; unset uses the ambient defaults when a
   *   backward attempt learns. An `api_key` entry needs `provider`; it authenticates the learner
   *   and is never recorded. A `base_url` entry needs `api_key` and is where the learner sends
   *   it; for `provider: azure` it is the required resource endpoint and needs no key.
   *   `organization`, `project` and `headers` entries need `api_key` and go with it.
   * @param options.maxChars Largest document the learner may write; unset uses
   *   `YIAN_PROFILE_MAX_CHARS` (32,000, about 8k tokens). Host writes are not bounded.
   * @param options.inputBudgetChars Character budget for episode JSON sent to the learner; unset
   *   uses `YIAN_PROFILE_INPUT_BUDGET_CHARS` (24,000). Feedback is always sent.
   * @param options.env The environment model defaults and credentials are read from; default
   *   `processEnv`.
   * @param options.createModel Build the learner's model from the config Python passes to
   *   `model_from_config`; default `chatModelFromConfig(config, {env})`.
   * @param options.onRetry Called before the learner's model call is retried; unset publishes
   *   nothing.
   * @throws TypeError when `store` lacks `read`/`write` or `llmConfig` isn't a record.
   * @throws DefaultsError when a budget is unset and a `YIAN_` variable holds a value it doesn't
   *   accept.
   * @throws RangeError when a budget isn't a positive integer, or the connection entries of
   *   `llmConfig` are invalid (see `popHostCredentials`).
   */
  constructor(
    store: MarkdownStore,
    options: {
      llmConfig?: JsonObject | null;
      maxChars?: number | bigint | null;
      inputBudgetChars?: number | bigint | null;
      env?: Env | null;
      createModel?: CreateModel | null;
      onRetry?: OnRetry | null;
    } = {},
  ) {
    const candidate = store as Partial<MarkdownStore> | null;
    if (typeof candidate?.read !== "function" || typeof candidate.write !== "function") {
      throw new TypeError("store must provide read() and write()");
    }
    const llmConfig = options.llmConfig ?? null;
    if (llmConfig !== null && !isRecord(llmConfig)) throw new TypeError("llm_config must be a record or null");
    // Defaults are read only for unset budgets, both resolved before either is checked, as Python does.
    const maxChars = options.maxChars ?? current().profileMaxChars;
    const inputBudgetChars = options.inputBudgetChars ?? current().profileInputBudgetChars;
    this.maxChars = budget(maxChars, "max_chars");
    this.inputBudgetChars = budget(inputBudgetChars, "input_budget_chars");
    this.store = store;
    this.llmConfig = fromHost(llmConfig ?? {}, "llm_config") as Record<string, unknown>;
    this.host = popHostCredentials(this.llmConfig);
    const env = options.env ?? processEnv;
    this.env = env;
    this.createModel = options.createModel ?? ((config) => chatModelFromConfig(config, { env }));
    this.onRetry = options.onRetry ?? undefined;
  }

  /** Reject later use of this backend and its sessions; repeated calls do nothing. No I/O. */
  close(): void {
    this.closed = true;
  }

  /**
   * Reject use after close.
   *
   * @throws MemoryOperationError `profile.closed` after `close()`.
   */
  private ensureOpen(stage: string): void {
    if (this.closed) throw error(stage, "profile.closed", "memory backend is closed");
  }

  /**
   * Return `[text, version]` from the store for host display or editing: the current document, or
   * the earlier `version`. Store errors pass through unchanged.
   *
   * @throws RangeError when the named version is unknown.
   * @throws MemoryOperationError `profile.closed` after `close()`.
   */
  async read(version: string | null = null): Promise<readonly [text: string, version: string]> {
    this.ensureOpen("read");
    return await this.store.read(version);
  }

  /**
   * Write a host or human edit with compare-and-set, without a budget check, and return the new
   * version. Store errors pass through unchanged.
   *
   * @throws MarkdownVersionMismatch when `baseVersion` is not current.
   * @throws TypeError or RangeError when the store rejects `provenance` as not JSON;
   *   `FileMarkdownStore` does so before writing anything.
   * @throws MemoryOperationError `profile.closed` after `close()`.
   */
  async write(text: string, options: { baseVersion: string; provenance?: StoreObject | null }): Promise<string> {
    this.ensureOpen("write");
    return await this.store.write(text, options);
  }

  /**
   * Refuse engine resets; delete or rewrite the profile through the store.
   *
   * @throws MemoryOperationError always: `profile.closed` after `close()`, else
   *   `profile.reset_unsupported`.
   */
  reset(): void {
    this.ensureOpen("reset");
    throw error("reset", "profile.reset_unsupported", "profile memory is reset through its store, not the engine");
  }

  /**
   * Pin the current document, or the recorded version when restoring.
   *
   * @param options.snapshot A snapshot recorded by an earlier session of this backend.
   * @throws MemoryOperationError `profile.closed` after `close()`, `profile.store_unavailable`
   *   (retryable) when the store can't be reached, `profile.store_invalid` when its document can't
   *   be decoded, or `profile.snapshot_unavailable` when the recorded version is malformed,
   *   unknown, or changed. Other store errors pass through unchanged.
   */
  async openRun(options: { snapshot?: JsonObject | null } = {}): Promise<ProfileSession> {
    this.ensureOpen("open_run");
    const snapshot = options.snapshot ?? null;
    let document: readonly [string, string];
    if (snapshot === null) {
      try {
        document = await this.store.read(null);
      } catch (cause) {
        throw readError("open_run", cause);
      }
      return new ProfileSession(this, document[0], document[1]);
    }
    try {
      if (
        !isRecord(snapshot) ||
        Object.keys(snapshot).length !== SNAPSHOT_KEYS.length ||
        !SNAPSHOT_KEYS.every((key) => Object.hasOwn(snapshot, key))
      ) {
        throw new RangeError("snapshot fields differ");
      }
      if (snapshot.format !== SNAPSHOT_FORMAT || snapshot.version !== SNAPSHOT_VERSION) {
        throw new RangeError("snapshot format differs");
      }
      const recorded = snapshot.profile_version;
      if (typeof recorded !== "string") throw new RangeError("profile_version must be a string");
      document = await this.store.read(recorded);
      if (document[1] !== recorded || sha256(document[0]) !== snapshot.text_sha256) {
        throw new RangeError("stored text differs from the recorded version");
      }
    } catch (cause) {
      if (!(cause instanceof TypeError || cause instanceof RangeError || isSystemError(cause))) throw cause;
      throw error("open_run", "profile.snapshot_unavailable", "recorded profile version cannot be restored", {
        cause,
      });
    }
    return new ProfileSession(this, document[0], document[1]);
  }
}

/**
 * One run's pinned view of the profile and its commit authority (Python's `_ProfileSession`).
 *
 * It reads its backend's private members through element access (`memory["store"]`), as Python's
 * session reads `self._memory._store`.
 */
export class ProfileSession implements MemorySession {
  /** The pinned version and text hash, never the text itself. Frozen. */
  readonly snapshot: JsonObject;
  private readonly memory: ProfileMemory;
  private readonly text: string;
  private readonly version: string;
  private readonly applied = new Map<string, readonly MemoryMutation[]>();
  // Commits run one at a time through this chain (Python's `Lock`).
  private queue: Promise<void> = Promise.resolve();
  private closed = false;

  /** @internal Pin one document; `ProfileMemory.openRun` builds sessions. */
  constructor(memory: ProfileMemory, text: string, version: string) {
    this.memory = memory;
    this.text = text;
    this.version = version;
    this.snapshot = Object.freeze({
      format: SNAPSHOT_FORMAT,
      version: SNAPSHOT_VERSION,
      profile_version: version,
      text_sha256: sha256(text),
    });
  }

  /** Mark the session closed; the store holds no per-session resources. */
  close(): void {
    this.closed = true;
  }

  /**
   * Reject use after this session or its backend closes.
   *
   * @throws MemoryOperationError `profile.closed` after the backend closes, else
   *   `profile.session_closed` after `close()`.
   */
  private ensureOpen(stage: string): void {
    this.memory["ensureOpen"](stage);
    if (this.closed) throw error(stage, "profile.session_closed", "profile session is closed");
  }

  /**
   * Return the full pinned document, never truncated; it does not depend on `context`.
   *
   * @throws MemoryOperationError `profile.invalid_context` for a context that isn't exactly a
   *   `RecallContext`; the session or backend is closed.
   */
  recall(context: RecallContext | null = null): string {
    this.ensureOpen("recall");
    if (context !== null && !isExactly(context, RecallContext)) {
      throw error("recall", "profile.invalid_context", "context must be a RecallContext");
    }
    return this.text;
  }

  /** Propose one full-document rewrite from the run's feedback. */
  learn(_context: LearnContext): readonly MemoryMutation[] {
    throw new Error("not implemented");
  }

  /**
   * Write the proposed document with compare-and-set; never call a model. Commits of one session
   * run one at a time, and a failed commit doesn't fail later ones. A repeated `commitId` with the
   * same batch is a no-op.
   *
   * @param mutations Empty, or the one document mutation this session proposed.
   * @throws MemoryOperationError `profile.version_mismatch` (retryable; nothing was written) when
   *   the document changed after the learner read it, `profile.store_unavailable` when the store
   *   failed during the write (it may have landed), `profile.store_invalid` when the store
   *   rejected it, or `profile.invalid_batch` for a malformed or reused batch; the session or
   *   backend is closed. Other store errors pass through unchanged.
   */
  commit(commitId: string, mutations: readonly MemoryMutation[]): Promise<void> {
    // Copied now, so a caller editing its array while earlier commits run can't change this batch.
    const batch = isList(mutations) ? [...mutations] : mutations;
    const run = this.queue.catch(() => {}).then(() => this.commitNow(commitId, batch));
    this.queue = run;
    return run;
  }

  /** One commit, run when the previous one has settled. */
  private async commitNow(commitId: unknown, mutations: unknown): Promise<void> {
    this.ensureOpen("commit");
    if (
      typeof commitId !== "string" ||
      !commitId ||
      !isList(mutations) ||
      !mutations.every((mutation) => isExactly(mutation, MemoryMutation))
    ) {
      throw error("commit", "profile.invalid_batch", "commit needs an id and a mutation array");
    }
    const batch: readonly MemoryMutation[] = Object.freeze([...mutations]);
    const applied = this.applied.get(commitId);
    if (applied !== undefined) {
      if (applied.length !== batch.length || !applied.every((mutation, i) => sameMemoryMutation(mutation, batch[i]!))) {
        throw error("commit", "profile.invalid_batch", "commit id was reused for another batch");
      }
      return;
    }
    if (batch.length === 0) {
      this.applied.set(commitId, batch);
      return;
    }
    const payload = batch.length === 1 ? batch[0]!.payload : null;
    if (!isRecord(payload) || typeof payload.text !== "string" || typeof payload.base_version !== "string") {
      throw error("commit", "profile.invalid_batch", "profile commits take at most one document mutation");
    }
    // The payload is frozen JSON; the store's JSON type differs only in having mutable arrays.
    const provenance = {
      commit_id: commitId,
      run_id: payload.run_id ?? null,
      seen_version: payload.seen_version ?? null,
      feedback: payload.feedback ?? null,
    } as StoreObject;
    try {
      await this.memory["store"].write(payload.text, { baseVersion: payload.base_version, provenance });
    } catch (cause) {
      if (cause instanceof MarkdownVersionMismatch) {
        throw error(
          "commit",
          "profile.version_mismatch",
          "profile changed after the learner read it; nothing was written",
          { retryable: true, cause },
        );
      }
      if (isSystemError(cause)) {
        throw error(
          "commit",
          "profile.store_unavailable",
          "profile store write failed and may have landed; check the store before retrying",
          { cause },
        );
      }
      if (cause instanceof RangeError) {
        throw error("commit", "profile.store_invalid", "profile store rejected the document", { cause });
      }
      throw cause;
    }
    this.applied.set(commitId, batch);
  }
}
