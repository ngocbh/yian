/**
 * Profile memory's backend, sessions and commits: a port of `tests/engine/test_profile_memory.py`.
 *
 * Ported: `test_constructor_validates_store_config_and_budgets`,
 * `test_host_read_and_write_use_the_store_without_a_budget`, `test_reset_is_refused`,
 * `test_closed_backend_rejects_later_use_without_touching_the_store`,
 * `test_closing_the_backend_invalidates_its_open_sessions`,
 * `test_closed_session_rejects_use_with_session_closed`, `test_snapshot_pins_version_and_hash_without_text`,
 * `test_restore_accepts_the_recorded_snapshot_after_later_edits`,
 * `test_restore_rejects_unknown_or_altered_snapshots`, `test_undecodable_document_is_a_permanent_open_failure`,
 * `test_recall_stays_pinned_after_a_later_write`, `test_restore_recalls_the_recorded_version_after_later_edits`,
 * `test_backend_satisfies_the_memory_protocols`, `test_commit_writes_with_compare_and_set_and_is_idempotent`,
 * `test_concurrent_write_after_learning_is_a_retryable_mismatch_without_a_model` and
 * `test_store_failure_during_commit_is_not_retryable`.
 *
 * TS-only: `bigint` budgets, a `0n` budget and the constructor's messages; a frozen snapshot; a
 * snapshot whose `version` is the `number` 1 is refused; an async host store (methods returning
 * promises) works throughout; a store `read` throwing a plain `Error` escapes unchanged from both
 * open paths, and one throwing a system error is `profile.store_unavailable` (retryable); recall
 * rejects a context that isn't a `RecallContext`; commit's batch messages; two concurrent commits
 * of one id write once; a failed commit (version mismatch) then a good one in the same session,
 * whose `2n` feedback is written as Python's int `2`.
 *
 * Changed:
 * - Every backend gets `baseEnv()`, an environment map standing in for Python's `conftest.py`
 *   variables, and each test starts from the built-in defaults.
 * - The commit tests and `test_closing_the_backend_invalidates_its_open_sessions` commit a
 *   hand-built document mutation with the payload `learn` proposes (`base_version`, `text`,
 *   `seen_version`, `run_id`, `feedback`) instead of learning it through a stub model; a
 *   `createModel` that throws stands in for patching `model_from_config` to fail.
 * - `LookupError` and `ValueError` from the store are `RangeError`; `OSError` is a Node system
 *   error (an `Error` with a string `code` and a `syscall`); the decode-error row of
 *   `test_store_failure_during_commit_is_not_retryable` is a `RangeError`.
 * - `test_constructor_validates_store_config_and_budgets`: budget errors are `RangeError`.
 * - `test_backend_satisfies_the_memory_protocols`: `isMemory`, and the session's methods are
 *   functions; the `implements` clauses are checked by the typecheck.
 * - `test_snapshot_pins_version_and_hash_without_text`: the snapshot's `version` is `1n`, and the
 *   hash is computed here, since `sha256` is private.
 * - `read`, `write`, `openRun` and `commit` are async; `with` blocks are `try`/`finally` around
 *   `close()`.
 *
 * Later: the learn rows (`session.learn` after close in
 * `test_closing_the_backend_invalidates_its_open_sessions`) and learned batches in the commit tests
 * arrive with learning.
 *
 * Not ported here: the reply-parsing, learning and credential tests, which arrive with learning.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BUILTIN, type Env, install } from "../../src/defaults.ts";
import {
  isMemory,
  type JsonObject,
  MemoryMutation,
  MemoryOperationError,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { ProfileMemory, type ProfileSession } from "../../src/memory/profile.ts";
import { FileMarkdownStore, InMemoryMarkdownStore, markdownVersion } from "../../src/memory/storage/markdown.ts";

// Cloned snapshots are edited freely.
type Mutable = any;

let home: string;

beforeEach(() => {
  install(BUILTIN);
  home = mkdtempSync(join(tmpdir(), "yian-profile-"));
});

afterEach(() => {
  install(BUILTIN);
  rmSync(home, { recursive: true, force: true });
});

/** Python's `conftest.py` environment: an empty private store and an explicit model default. */
function baseEnv(overrides: Record<string, string> = {}): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
    ...Object.entries(overrides),
  ]);
  return (name) => vars.get(name);
}

const RECALL = new RecallContext("system", "Say hi.", {}, ["chat", "reply"]);
const CONFIG = { provider: "ollama", model: "local-profiler" };

/** The hex SHA-256 of one UTF-8 string. */
function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** A profile memory over `<home>/ada.md` seeded with `text` (Python's `_memory`). */
async function newMemory(
  text = "# Ada\n",
  options: ConstructorParameters<typeof ProfileMemory>[1] = {},
): Promise<ProfileMemory> {
  const store = new FileMarkdownStore(join(home, "ada.md"));
  await store.write(text, { baseVersion: (await store.read())[1] });
  return new ProfileMemory(store, { llmConfig: CONFIG, env: baseEnv(), ...options });
}

/** A model factory that fails the test if anything builds a model. */
function noModel(): never {
  throw new Error("commit built a model");
}

/** The document mutation `learn` proposes: `text` over `baseVersion`, seen at `seenVersion`. */
function documentMutation(
  baseVersion: string,
  text: string,
  { seenVersion = baseVersion, feedback = ["x"] as unknown[] } = {},
): MemoryMutation {
  return new MemoryMutation(`profile:${sha256(text)}`, {
    base_version: baseVersion,
    text,
    seen_version: seenVersion,
    run_id: "run-1",
    feedback,
  });
}

/** Open a run, await `fn` with it and close it (Python's `with memory.open_run() as session`). */
async function withRun<T>(
  memory: ProfileMemory,
  fn: (session: ProfileSession) => T | Promise<T>,
  snapshot?: JsonObject,
): Promise<T> {
  const session = await memory.openRun({ snapshot });
  try {
    return await fn(session);
  } finally {
    session.close();
  }
}

/** Return the `MemoryOperationError` `fn` throws or rejects with. */
async function failure(fn: () => unknown): Promise<MemoryOperationError> {
  try {
    await fn();
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

/** The lines of the document's write log. */
function writeLog(): string[] {
  return readFileSync(join(home, ".ada.md.writes.jsonl"), "utf8").trimEnd().split("\n");
}

/** A Node system error, as storage throws when it can't be reached. */
function systemError(message: string): Error {
  return Object.assign(new Error(message), { code: "EIO", syscall: "write" });
}

test("test_constructor_validates_store_config_and_budgets", () => {
  const store = new FileMarkdownStore(join(home, "ada.md"));
  for (const bad of [{}, null, { read() {}, write: "x" }]) {
    expect(() => new ProfileMemory(bad as Mutable)).toThrow(new TypeError("store must provide read() and write()"));
  }
  for (const bad of ["ollama", ["ollama"]]) {
    expect(() => new ProfileMemory(store, { llmConfig: bad as Mutable })).toThrow(
      new TypeError("llm_config must be a record or null"),
    );
  }
  const rows: [Mutable, string][] = [
    [{ maxChars: 0 }, "max_chars"],
    [{ inputBudgetChars: -1 }, "input_budget_chars"],
    [{ maxChars: 1.5 }, "max_chars"],
    [{ maxChars: 0n }, "max_chars"],
    [{ maxChars: true }, "max_chars"],
    [{ inputBudgetChars: 2n ** 60n }, "input_budget_chars"],
  ];
  for (const [budget, name] of rows) {
    expect(() => new ProfileMemory(store, { env: baseEnv(), ...budget })).toThrow(
      new RangeError(`${name} must be a positive integer`),
    );
  }
  const memory = new ProfileMemory(store, { env: baseEnv() });
  expect([memory.maxChars, memory.inputBudgetChars]).toEqual([32_000, 24_000]);
  const big = new ProfileMemory(store, { env: baseEnv(), maxChars: 10n, inputBudgetChars: 20n, llmConfig: null });
  expect([big.maxChars, big.inputBudgetChars]).toEqual([10, 20]);
});

test("test_host_read_and_write_use_the_store_without_a_budget", async () => {
  const memory = await newMemory("# Ada\n", { maxChars: 10 });
  const [text, version] = await memory.read();
  const longText = `# Ada\n${"x".repeat(100)}\n`;
  const newVersion = await memory.write(longText, { baseVersion: version, provenance: { by: "ada" } });
  expect(await memory.read()).toEqual([longText, newVersion]);
  expect(await memory.read(version)).toEqual([text, version]);
});

test("test_reset_is_refused", async () => {
  const memory = await newMemory();
  const error = await failure(() => memory.reset());
  expect([error.stage, error.code, error.retryable]).toEqual(["reset", "profile.reset_unsupported", false]);
});

test("test_closed_backend_rejects_later_use_without_touching_the_store", async () => {
  const memory = new ProfileMemory(new FileMarkdownStore(join(home, "ada.md")), { env: baseEnv() });
  memory.close();
  memory.close();
  const calls = [
    () => memory.openRun(),
    () => memory.reset(),
    () => memory.read(),
    () => memory.write("x", { baseVersion: markdownVersion("") }),
  ];
  for (const call of calls) {
    const error = await failure(call);
    expect(error.code).toBe("profile.closed");
    expect(error.retryable).toBe(false);
  }
  expect(readdirSync(home)).toEqual([]);
});

test("test_closing_the_backend_invalidates_its_open_sessions", async () => {
  const memory = await newMemory();
  const session = await memory.openRun();
  const version = (await memory.read())[1];
  const batch = [documentMutation(version, "# Ada\nBrief.\n", { feedback: ["Be brief."] })];
  memory.close();
  for (const call of [() => session.recall(RECALL), () => session.commit("commit-1", batch)]) {
    expect((await failure(call)).code).toBe("profile.closed");
  }
  expect(readFileSync(join(home, "ada.md"), "utf8")).toBe("# Ada\n");
});

test("test_closed_session_rejects_use_with_session_closed", async () => {
  const memory = await newMemory();
  const session = await memory.openRun();
  session.close();
  const error = await failure(() => session.recall(RECALL));
  expect(error.code).toBe("profile.session_closed");
  expect(error.retryable).toBe(false);
  (await memory.openRun()).recall(RECALL);
});

test("test_snapshot_pins_version_and_hash_without_text", async () => {
  const memory = await newMemory();
  const [text, version] = await memory.read();
  await withRun(memory, (session) => {
    expect(session.snapshot).toEqual({
      format: "yian.profile_memory",
      version: 1n,
      profile_version: version,
      text_sha256: sha256(text),
    });
    expect(Object.isFrozen(session.snapshot)).toBe(true);
  });
});

test("test_restore_accepts_the_recorded_snapshot_after_later_edits", async () => {
  const memory = await newMemory();
  const recorded = await withRun(memory, (session) => session.snapshot);
  await memory.write("# Ada\nLater.\n", { baseVersion: (await memory.read())[1] });
  await withRun(memory, (restored) => expect(restored.snapshot).toEqual(recorded), recorded);
});

test("test_restore_rejects_unknown_or_altered_snapshots", async () => {
  const memory = await newMemory();
  const recorded = await withRun(memory, (session) => session.snapshot);
  const changes: Mutable[] = [
    { profile_version: `sha256:${"0".repeat(64)}` },
    { text_sha256: "0".repeat(64) },
    { format: "other" },
    { extra: 1n },
    { version: 1 },
  ];
  for (const change of changes) {
    const error = await failure(() => memory.openRun({ snapshot: { ...recorded, ...change } }));
    expect([error.code, error.message]).toEqual([
      "profile.snapshot_unavailable",
      "recorded profile version cannot be restored",
    ]);
    expect(error.cause).toBeInstanceOf(RangeError);
  }
});

test("test_undecodable_document_is_a_permanent_open_failure", async () => {
  const memory = await newMemory();
  writeFileSync(join(home, "ada.md"), Buffer.from("# Ada \xff\n", "latin1"));
  const error = await failure(() => memory.openRun());
  expect([error.code, error.retryable]).toEqual(["profile.store_invalid", false]);
  expect(error.message).toBe("profile store returned an unreadable document");
});

test("test_recall_stays_pinned_after_a_later_write", async () => {
  const memory = await newMemory();
  await withRun(memory, async (session) => {
    await memory.write("# Ada\nChanged.\n", { baseVersion: (await memory.read())[1] });
    expect(session.recall(RECALL)).toBe("# Ada\n");
  });
});

test("test_restore_recalls_the_recorded_version_after_later_edits", async () => {
  const memory = await newMemory();
  const recorded = await withRun(memory, (session) => session.snapshot);
  await memory.write("# Ada\nLater.\n", { baseVersion: (await memory.read())[1] });
  await withRun(memory, (restored) => expect(restored.recall(RECALL)).toBe("# Ada\n"), recorded);
});

test("test_backend_satisfies_the_memory_protocols", async () => {
  const memory = await newMemory();
  await withRun(memory, (session) => {
    expect(isMemory(memory)).toBe(true);
    for (const method of [session.recall, session.learn, session.commit, session.close]) {
      expect(typeof method).toBe("function");
    }
  });
});

test("recall rejects a context that isn't a RecallContext", async () => {
  const memory = await newMemory();
  await withRun(memory, async (session) => {
    expect(session.recall()).toBe("# Ada\n");
    const error = await failure(() => session.recall({ ...RECALL } as Mutable));
    expect([error.stage, error.code, error.message]).toEqual([
      "recall",
      "profile.invalid_context",
      "context must be a RecallContext",
    ]);
  });
});

test("test_commit_writes_with_compare_and_set_and_is_idempotent", async () => {
  const memory = await newMemory("# Ada\n", { createModel: noModel });
  await withRun(memory, async (session) => {
    const version = (await memory.read())[1];
    const batch = [documentMutation(version, "# Ada\nBrief.\n", { feedback: ["Be brief."] })];
    await session.commit("commit-1", batch);
    await session.commit("commit-1", batch);
    await session.commit("commit-empty", []);
    const error = await failure(() => session.commit("commit-1", []));
    expect(error.code).toBe("profile.invalid_batch");
  });
  expect((await memory.read())[0]).toBe("# Ada\nBrief.\n");
  const writes = writeLog();
  expect(writes.length).toBe(2);
  const provenance = JSON.parse(writes.at(-1)!).provenance;
  expect([provenance.commit_id, provenance.run_id]).toEqual(["commit-1", "run-1"]);
  expect(Object.keys(provenance).sort()).toEqual(["commit_id", "feedback", "run_id", "seen_version"]);
  expect(provenance.feedback).toEqual(["Be brief."]);
});

test("test_concurrent_write_after_learning_is_a_retryable_mismatch_without_a_model", async () => {
  const memory = await newMemory("# Ada\n", { createModel: noModel });
  const error = await withRun(memory, async (session) => {
    const batch = [documentMutation((await memory.read())[1], "# Ada\nFrom model.\n")];
    await memory.write("# Ada\nFrom human.\n", { baseVersion: (await memory.read())[1] });
    return failure(() => session.commit("commit-1", batch));
  });
  expect([error.stage, error.code, error.retryable]).toEqual(["commit", "profile.version_mismatch", true]);
  expect((await memory.read())[0]).toBe("# Ada\nFrom human.\n");
});

test("test_store_failure_during_commit_is_not_retryable", async () => {
  const store = new FileMarkdownStore(join(home, "ada.md"));
  await store.write("# Ada\n", { baseVersion: (await store.read())[1] });
  const memory = new ProfileMemory(store, { llmConfig: CONFIG, env: baseEnv() });
  await withRun(memory, async (session) => {
    const batch = [documentMutation((await memory.read())[1], "# Ada\nNew.\n")];
    const rows: [Error, string, string][] = [
      [
        systemError("disk"),
        "profile.store_unavailable",
        "profile store write failed and may have landed; check the store before retrying",
      ],
      [new RangeError("document is not valid UTF-8"), "profile.store_invalid", "profile store rejected the document"],
    ];
    for (const [thrown, code, message] of rows) {
      store.write = async () => {
        throw thrown;
      };
      const error = await failure(() => session.commit(`commit-${code}`, batch));
      expect([error.code, error.retryable, error.message]).toEqual([code, false, message]);
      expect(error.cause).toBe(thrown);
    }
  });
});

test("commit's batch messages", async () => {
  const memory = await newMemory();
  await withRun(memory, async (session) => {
    const version = (await memory.read())[1];
    const mutation = documentMutation(version, "# Ada\nBrief.\n");
    const rows: [string, Mutable, string][] = [
      ["", [], "commit needs an id and a mutation array"],
      ["commit-1", mutation, "commit needs an id and a mutation array"],
      ["commit-1", [{ ...mutation }], "commit needs an id and a mutation array"],
      ["commit-1", [mutation, mutation], "profile commits take at most one document mutation"],
      ["commit-1", [new MemoryMutation("m", ["x"])], "profile commits take at most one document mutation"],
      ["commit-1", [new MemoryMutation("m", { text: "x" })], "profile commits take at most one document mutation"],
    ];
    for (const [commitId, mutations, message] of rows) {
      const error = await failure(() => session.commit(commitId, mutations));
      expect([error.stage, error.code, error.message]).toEqual(["commit", "profile.invalid_batch", message]);
    }
    await session.commit("commit-1", []);
    const error = await failure(() => session.commit("commit-1", [mutation]));
    expect(error.message).toBe("commit id was reused for another batch");
  });
});

test("an async host store works throughout", async () => {
  const inner = new InMemoryMarkdownStore("# Ada\n");
  const store = {
    read: async (version?: string | null) => inner.read(version),
    write: async (text: string, options: { baseVersion: string }) => inner.write(text, options),
  };
  const memory = new ProfileMemory(store, { env: baseEnv() });
  const [text, version] = await memory.read();
  expect(text).toBe("# Ada\n");
  const recorded = await withRun(memory, async (session) => {
    expect(session.recall(RECALL)).toBe("# Ada\n");
    await session.commit("commit-1", [documentMutation(version, "# Ada\nBrief.\n")]);
    return session.snapshot;
  });
  expect((await memory.read())[0]).toBe("# Ada\nBrief.\n");
  await withRun(memory, (restored) => expect(restored.recall()).toBe("# Ada\n"), recorded);
  const error = await withRun(memory, (session) =>
    failure(() => session.commit("commit-1", [documentMutation(version, "# Ada\nAgain.\n")])),
  );
  expect(error.code).toBe("profile.version_mismatch");
});

test("store read failures: a plain Error escapes, a system error is retryable", async () => {
  const sentinel = new Error("sentinel");
  let thrown: Error = sentinel;
  const store = {
    read: () => {
      throw thrown;
    },
    write: () => "unused",
  };
  const memory = new ProfileMemory(store, { env: baseEnv() });
  const snapshot = { format: "yian.profile_memory", version: 1n, profile_version: "v", text_sha256: "0" };
  for (const options of [{}, { snapshot }]) {
    await expect(memory.openRun(options)).rejects.toBe(sentinel);
  }
  thrown = systemError("down");
  const error = await failure(() => memory.openRun());
  expect([error.code, error.retryable, error.message]).toEqual([
    "profile.store_unavailable",
    true,
    "profile store could not be read",
  ]);
  expect(error.cause).toBe(thrown);
  expect((await failure(() => memory.openRun({ snapshot }))).code).toBe("profile.snapshot_unavailable");
});

test("two concurrent commits of one id write once", async () => {
  const memory = await newMemory();
  await withRun(memory, async (session) => {
    const batch = [documentMutation((await memory.read())[1], "# Ada\nBrief.\n")];
    await Promise.all([session.commit("commit-1", batch), session.commit("commit-1", batch)]);
  });
  expect(writeLog().length).toBe(2);
});

test("a failed commit then a good one in the same session; bigint feedback writes as an int", async () => {
  const memory = await newMemory();
  await withRun(memory, async (session) => {
    const stale = [documentMutation((await memory.read())[1], "# Ada\nFrom model.\n")];
    await memory.write("# Ada\nFrom human.\n", { baseVersion: (await memory.read())[1] });
    const fresh = [documentMutation((await memory.read())[1], "# Ada\nBoth.\n", { feedback: [2n] })];
    const results = await Promise.allSettled([session.commit("commit-1", stale), session.commit("commit-2", fresh)]);
    expect(results.map((result) => result.status)).toEqual(["rejected", "fulfilled"]);
  });
  expect((await memory.read())[0]).toBe("# Ada\nBoth.\n");
  const line = writeLog().at(-1)!;
  expect(line).toContain('"feedback": [2]');
  expect(JSON.parse(line).provenance.feedback).toEqual([2]);
});
