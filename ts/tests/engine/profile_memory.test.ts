/**
 * Profile memory's backend, sessions, learning and commits: a port of
 * `tests/engine/test_profile_memory.py`.
 *
 * Ported: `test_constructor_validates_store_config_and_budgets`,
 * `test_host_read_and_write_use_the_store_without_a_budget`, `test_reset_is_refused`,
 * `test_closed_backend_rejects_later_use_without_touching_the_store`,
 * `test_closing_the_backend_invalidates_its_open_sessions`,
 * `test_closed_session_rejects_use_with_session_closed`, `test_snapshot_pins_version_and_hash_without_text`,
 * `test_restore_accepts_the_recorded_snapshot_after_later_edits`,
 * `test_restore_rejects_unknown_or_altered_snapshots`, `test_undecodable_document_is_a_permanent_open_failure`,
 * `test_recall_stays_pinned_after_a_later_write`, `test_restore_recalls_the_recorded_version_after_later_edits`,
 * `test_parse_reply_enforces_tags_content_and_budget`,
 * `test_parse_reply_joins_text_blocks_and_ends_with_one_newline`, `test_learn_without_feedback_calls_no_model`,
 * `test_learn_rewrites_the_current_document_from_feedback_and_episodes`,
 * `test_learner_sees_the_tool_calls_of_each_episode`,
 * `test_learn_keeps_only_the_newest_episodes_within_the_input_budget`,
 * `test_an_oversized_newest_episode_is_trimmed_not_dropped`,
 * `test_provider_failure_is_retryable_and_bad_replies_are_not`, `test_unchanged_document_proposes_nothing`,
 * `test_backend_satisfies_the_memory_protocols`, `test_commit_writes_with_compare_and_set_and_is_idempotent`,
 * `test_concurrent_write_after_learning_is_a_retryable_mismatch_without_a_model`,
 * `test_store_failure_during_commit_is_not_retryable`, `test_learner_errors_chain_a_cause_without_the_key`,
 * `test_learner_errors_scrub_the_key_variable_and_saved_login` and
 * `test_session_recalls_and_learns_the_whole_document`.
 *
 * TS-only: `bigint` budgets, a `0n` budget and the constructor's messages; a `bigint` budget above
 * 2^53 − 1 is refused (budgets are numbers); a bad `YIAN_PROFILE_MAX_CHARS` is read only for an
 * unset budget; an array subclass is not a mutation array; a frozen snapshot; a
 * snapshot whose `version` is the `number` 1 is refused; an async host store (methods returning
 * promises) works throughout; a store `read` throwing a plain `Error` escapes unchanged from both
 * open paths, and one throwing a system error is `profile.store_unavailable` (retryable), on open
 * and during learn; recall rejects a context that isn't a `RecallContext`; commit's batch
 * messages; two concurrent commits of one id write once; a failed commit (version mismatch) then a
 * good one in the same session, whose `2n` feedback is written as Python's int `2`; the reply
 * parser's messages; feedback holding `Infinity` is `profile.invalid_context` after the model call;
 * `onRetry` sees a retried `ProviderError`; a commit id whose write failed commits once the store
 * is back; with `YIAN_TEST_NODE` set to a Node binary, the built backend opens a document Bun
 * seeded, learns with a scripted model and commits under Node, and Bun reads the new document and
 * its write log.
 *
 * Changed:
 * - Every backend gets `baseEnv()`, an environment map standing in for Python's `conftest.py`
 *   variables, and each test starts from the built-in defaults.
 * - `installModel(replies)` stands in for Python's `_install`: the backend's `createModel` option
 *   records each config and returns a `ScriptedModel` answering with the next reply of a shared
 *   list, a text `Response` or an `Error`. Where Python patches `model_from_config` to fail after
 *   learning, the test checks that no further config was recorded; a `createModel` that throws
 *   stands in for patching it to fail.
 * - The TS-only commit tests commit a hand-built document mutation with the payload `learn`
 *   proposes (`base_version`, `text`, `seen_version`, `run_id`, `feedback`).
 * - `test_parse_reply_enforces_tags_content_and_budget`: the `object()` row is a reply with no
 *   text part, which reads as `""` and so fails on the tags.
 * - `test_parse_reply_joins_text_blocks_and_ends_with_one_newline`: the blocks are two text parts.
 * - `test_session_recalls_and_learns_the_whole_document`: `openFrame` and `frameRef` stand in for
 *   `open_frame` and `frame_ref`.
 * - `test_learner_errors_scrub_the_key_variable_and_saved_login`: an `env` map without defaults and
 *   a saved login through `AuthStore` stand in for the process environment.
 * - `LookupError` and `ValueError` from the store are `RangeError`; `OSError` is a Node system
 *   error (an `Error` with a string `code` and a `syscall`); the decode-error row of
 *   `test_store_failure_during_commit_is_not_retryable` is a `RangeError`.
 * - `test_constructor_validates_store_config_and_budgets`: budget errors are `RangeError`.
 * - `test_backend_satisfies_the_memory_protocols`: `isMemory`, and the session's methods are
 *   functions; the `implements` clauses are checked by the typecheck.
 * - `test_snapshot_pins_version_and_hash_without_text`: the snapshot's `version` is `1n`, and the
 *   hash is computed here, since `sha256` is private.
 * - `test_an_oversized_newest_episode_is_trimmed_not_dropped`: `_canonical_json`'s length is the
 *   code-point count of `writeAsJson`.
 * - JSON ints in feedback are `bigint`s; `read`, `write`, `openRun`, `learn` and `commit` are
 *   async; `with` blocks are `try`/`finally` around `close()`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { BUILTIN, DefaultsError, type Env, forget, install } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import type { RetryInfo } from "../../src/llm/executor.ts";
import { Message, type Request, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ApiKeyCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import {
  isMemory,
  type JsonObject,
  LearnContext,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOperationError,
  ModelTextInteraction,
  RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
  type VisibleInteraction,
} from "../../src/memory/contracts.ts";
import { parseReply, ProfileMemory, type ProfileSession } from "../../src/memory/profile.ts";
import { FileMarkdownStore, InMemoryMarkdownStore, markdownVersion } from "../../src/memory/storage/markdown.ts";
import { type Value, writeAsJson } from "../../src/typesys/values.ts";

// Rows pass values the signatures refuse, to reach the runtime checks.
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

/** A scripted assistant reply whose text is `text`. */
function textReply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });
}

/**
 * Python's `_install`: a `createModel` option that records each config and returns a new
 * `ScriptedModel` (named by the config) answering with the next reply of the shared list, given
 * as text or an `Error`.
 */
function installModel(replies: readonly (string | Error)[]) {
  const queue = replies.map((reply) => (typeof reply === "string" ? textReply(reply) : reply));
  const configs: Record<string, unknown>[] = [];
  const models: ScriptedModel[] = [];
  const createModel = (config: Readonly<Record<string, unknown>>): ScriptedModel => {
    configs.push(structuredClone(config) as Record<string, unknown>);
    const next = queue.shift();
    const model = new ScriptedModel(next === undefined ? [] : [next], {
      provider: String(config.provider),
      model: String(config.model),
    });
    models.push(model);
    return model;
  };
  return {
    createModel,
    configs,
    /** Every request the models received, in order. */
    get requests(): Request[] {
      return models.flatMap((model) => model.requests);
    },
  };
}

/** Python's `_feedback`: one output feedback entry per value. */
function feedbackOf(...values: unknown[]): Record<string, unknown>[] {
  return values.map((value) => ({
    channel: "output",
    target_name: null,
    value,
    owner_static_path: null,
    execution_seq: null,
  }));
}

/** One succeeded AGENT episode of `chat.reply` (Python's `_episode`). */
function episodeOf(prompt: string, output: string, interactions: readonly VisibleInteraction[] = []): MemoryEpisode {
  return new MemoryEpisode("run-1", [], ["chat", "reply"], MemoryEpisodeStatus.SUCCEEDED, { topic: prompt }, {
    output,
    interactions,
    baseSystemPrompt: "system",
    prompt,
  });
}

/** A learning context of `run-1` for the `profile` memory. */
function learnContext(episodes: readonly MemoryEpisode[], feedback: readonly unknown[] = []): LearnContext {
  return new LearnContext("run-1", "profile", episodes, feedback);
}

/** The learner's JSON input in one recorded request, the last by default. */
function requestMaterial(model: { requests: Request[] }, index = -1): Mutable {
  return JSON.parse(model.requests.at(index)!.messages[1]!.text());
}

/** The number of code points in `text` (Python's `len`). */
function codePoints(text: string): number {
  return Array.from(text).length;
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
    expect(() => new ProfileMemory(bad as Mutable, { env: baseEnv() })).toThrow(
      new TypeError("store must provide read() and write()"),
    );
  }
  for (const bad of ["ollama", ["ollama"]]) {
    expect(() => new ProfileMemory(store, { env: baseEnv(), llmConfig: bad as Mutable })).toThrow(
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

test("a bad defaults variable is read only for an unset budget", () => {
  const store = new FileMarkdownStore(join(home, "ada.md"));
  const variable = "YIAN_PROFILE_MAX_CHARS";
  const saved = process.env[variable];
  process.env[variable] = "0";
  forget();
  try {
    const memory = new ProfileMemory(store, { env: baseEnv(), maxChars: 10, inputBudgetChars: 20 });
    expect([memory.maxChars, memory.inputBudgetChars]).toEqual([10, 20]);
    expect(() => new ProfileMemory(store, { env: baseEnv(), inputBudgetChars: 20 })).toThrow(DefaultsError);
  } finally {
    if (saved === undefined) delete process.env[variable];
    else process.env[variable] = saved;
  }
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
  const model = installModel(["<profile># Ada\nBrief.</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  const session = await memory.openRun();
  const batch = await session.learn(learnContext([], feedbackOf("Be brief.")));
  memory.close();
  const calls = [
    () => session.recall(RECALL),
    () => session.learn(learnContext([], feedbackOf("Be brief."))),
    () => session.commit("commit-1", batch),
  ];
  for (const call of calls) {
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

test("test_parse_reply_enforces_tags_content_and_budget", () => {
  const noText = new Response({ message: new Message({ role: "assistant", parts: [] }) });
  const noTags = "profile learner reply must wrap the document in <profile> tags";
  const rows: [Response, string, string][] = [
    [textReply("# Ada without tags"), "profile.contract", noTags],
    [textReply("<profile>   </profile>"), "profile.contract", "profile learner returned an empty document"],
    [
      textReply(`<profile>${"y".repeat(101)}</profile>`),
      "profile.over_budget",
      "profile learner returned 101 characters; the limit is 100",
    ],
    [noText, "profile.contract", noTags],
  ];
  for (const [reply, code, message] of rows) {
    let caught: unknown;
    try {
      parseReply(reply, "# Ada\n", 100);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(MemoryOperationError);
    const error = caught as MemoryOperationError;
    expect([error.stage, error.code, error.retryable, error.message]).toEqual(["learn", code, false, message]);
  }
});

test("test_parse_reply_joins_text_blocks_and_ends_with_one_newline", () => {
  const reply = new Response({
    message: new Message({
      role: "assistant",
      parts: [new TextPart("Here:\n<profile>\n# Ada\n"), new TextPart("- Brief.\n</profile>")],
    }),
  });
  expect(parseReply(reply, "# Ada\n", 100)).toBe("# Ada\n- Brief.\n");
  expect(parseReply(textReply("<profile></profile>"), "", 100)).toBe("");
});

test("test_learn_without_feedback_calls_no_model", async () => {
  const createModel = (): never => {
    throw new Error("model constructed");
  };
  const memory = await newMemory("# Ada\n", { createModel });
  await withRun(memory, async (session) => {
    expect(await session.learn(learnContext([episodeOf("hi", "hello")]))).toEqual([]);
  });
});

test("test_learn_rewrites_the_current_document_from_feedback_and_episodes", async () => {
  const model = installModel(["<profile>\n# Ada\n- Prefers brief replies.\n</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  let seenVersion = "";
  let currentVersion = "";
  const [mutation, ...rest] = await withRun(memory, async (session) => {
    seenVersion = (await memory.read())[1];
    await memory.write("# Ada\nHuman note.\n", { baseVersion: seenVersion });
    currentVersion = (await memory.read())[1];
    const call = new ToolCallInteraction("calendar", { day: "today" });
    const episodes = [episodeOf("first", "long answer", [call]), episodeOf("second", "longer answer")];
    return session.learn(learnContext(episodes, feedbackOf("Be brief.", "Be brief.")));
  });
  expect(rest).toEqual([]);
  // An ollama config resolves to itself.
  expect(model.configs).toEqual([CONFIG]);
  const material = requestMaterial(model);
  expect(material.current_profile).toBe("# Ada\nHuman note.\n");
  expect(material.seen_profile).toBe("# Ada\n");
  expect(material.feedback.map((entry: Mutable) => entry.value)).toEqual(["Be brief.", "Be brief."]);
  expect(material.episodes.map((episode: Mutable) => episode.prompt)).toEqual(["first", "second"]);
  expect(material.episodes[0].interactions).toEqual([
    { kind: "tool_call", name: "calendar", arguments: { day: "today" } },
  ]);
  expect(material.omitted_episodes).toBe(0);
  expect(mutation!.payload).toEqual({
    base_version: currentVersion,
    text: "# Ada\n- Prefers brief replies.\n",
    seen_version: seenVersion,
    run_id: "run-1",
    feedback: ["Be brief."],
  });
});

test("test_learner_sees_the_tool_calls_of_each_episode", async () => {
  const model = installModel(["<profile># Ada\n- Check the calendar first.</profile>"]);
  const interactions = [
    new ModelTextInteraction("Let me look."),
    new ToolCallInteraction("calendar", { day: "friday" }),
    new ToolResultInteraction("calendar", "busy 9-17"),
  ];
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  const context = learnContext(
    [episodeOf("free friday?", "Yes.", interactions)],
    feedbackOf("Check the calendar before answering."),
  );
  await withRun(memory, (session) => session.learn(context));
  const { episodes } = requestMaterial(model);
  expect(episodes.length).toBe(1);
  expect(episodes[0].interactions).toEqual([
    { kind: "model_text", text: "Let me look." },
    { kind: "tool_call", name: "calendar", arguments: { day: "friday" } },
    { kind: "tool_result", name: "calendar", text: "busy 9-17" },
  ]);
});

test("test_learn_keeps_only_the_newest_episodes_within_the_input_budget", async () => {
  const model = installModel(["<profile># Ada\nNew.</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel, inputBudgetChars: 700 });
  const episodes = [0, 1, 2, 3, 4].map((index) => episodeOf(`prompt ${index}`, "o".repeat(100)));
  await withRun(memory, (session) => session.learn(learnContext(episodes, feedbackOf("x"))));
  const material = requestMaterial(model);
  expect(material.episodes.map((episode: Mutable) => episode.prompt)).toEqual(["prompt 3", "prompt 4"]);
  expect(material.omitted_episodes).toBe(3);
});

test("test_an_oversized_newest_episode_is_trimmed_not_dropped", async () => {
  const model = installModel(["<profile># Ada\nRead less.</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  const big = [
    new ToolCallInteraction("read_file", { path: "notes.md" }),
    new ToolResultInteraction("read_file", "x".repeat(30_000)),
  ];
  const episodes = [episodeOf("older", "short"), episodeOf("newest", "done", big)];
  await withRun(memory, (session) => session.learn(learnContext(episodes, feedbackOf("x"))));
  const material = requestMaterial(model);
  expect(material.omitted_episodes).toBe(1);
  expect(material.episodes.length).toBe(1);
  const [newest] = material.episodes;
  expect(newest.truncated).toBe(true);
  expect(newest.original_chars).toBeGreaterThan(memory.inputBudgetChars);
  expect(newest.prefix).toContain('"name":"read_file"');
  expect(newest.suffix).toContain('"prompt":"newest"');
  expect(codePoints(writeAsJson(newest as Value))).toBeLessThanOrEqual(memory.inputBudgetChars);
});

test("test_provider_failure_is_retryable_and_bad_replies_are_not", async () => {
  const model = installModel([new Error("provider down"), "no tags"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  await withRun(memory, async (session) => {
    for (const [code, retryable] of [
      ["profile.provider", true],
      ["profile.contract", false],
    ] as const) {
      const error = await failure(() => session.learn(learnContext([], feedbackOf("x"))));
      expect([error.code, error.retryable]).toEqual([code, retryable]);
    }
  });
});

test("test_unchanged_document_proposes_nothing", async () => {
  const model = installModel(["<profile># Ada</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  await withRun(memory, async (session) => {
    expect(await session.learn(learnContext([], feedbackOf("x")))).toEqual([]);
  });
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
  const model = installModel(["<profile># Ada\nBrief.</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  await withRun(memory, async (session) => {
    const batch = await session.learn(learnContext([], feedbackOf("Be brief.")));
    await session.commit("commit-1", batch);
    await session.commit("commit-1", batch);
    await session.commit("commit-empty", []);
    const error = await failure(() => session.commit("commit-1", []));
    expect(error.code).toBe("profile.invalid_batch");
  });
  // Commit built no model.
  expect(model.configs.length).toBe(1);
  expect((await memory.read())[0]).toBe("# Ada\nBrief.\n");
  const writes = writeLog();
  expect(writes.length).toBe(2);
  const provenance = JSON.parse(writes.at(-1)!).provenance;
  expect([provenance.commit_id, provenance.run_id]).toEqual(["commit-1", "run-1"]);
  expect(Object.keys(provenance).sort()).toEqual(["commit_id", "feedback", "run_id", "seen_version"]);
  expect(provenance.feedback).toEqual(["Be brief."]);
});

test("test_concurrent_write_after_learning_is_a_retryable_mismatch_without_a_model", async () => {
  const model = installModel(["<profile># Ada\nFrom model.</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  const error = await withRun(memory, async (session) => {
    const batch = await session.learn(learnContext([], feedbackOf("x")));
    await memory.write("# Ada\nFrom human.\n", { baseVersion: (await memory.read())[1] });
    return failure(() => session.commit("commit-1", batch));
  });
  expect([error.stage, error.code, error.retryable]).toEqual(["commit", "profile.version_mismatch", true]);
  expect(model.configs.length).toBe(1);
  expect((await memory.read())[0]).toBe("# Ada\nFrom human.\n");
});

test("test_store_failure_during_commit_is_not_retryable", async () => {
  const store = new FileMarkdownStore(join(home, "ada.md"));
  await store.write("# Ada\n", { baseVersion: (await store.read())[1] });
  const model = installModel(["<profile># Ada\nNew.</profile>"]);
  const memory = new ProfileMemory(store, { llmConfig: CONFIG, env: baseEnv(), createModel: model.createModel });
  await withRun(memory, async (session) => {
    const batch = await session.learn(learnContext([], feedbackOf("x")));
    const rows: [Error, string, string][] = [
      [
        systemError("disk"),
        "profile.store_unavailable",
        "profile store write failed and may have landed; check the store before retrying",
      ],
      [new RangeError("document is not valid UTF-8"), "profile.store_invalid", "profile store rejected the document"],
    ];
    const write = store.write;
    for (const [thrown, code, message] of rows) {
      store.write = async () => {
        throw thrown;
      };
      const error = await failure(() => session.commit(`commit-${code}`, batch));
      expect([error.code, error.retryable, error.message]).toEqual([code, false, message]);
      expect(error.cause).toBe(thrown);
    }
    // A failed write records nothing, so the same id commits once the store is back.
    store.write = write;
    const before = writeLog().length;
    await session.commit("commit-profile.store_unavailable", batch);
    await session.commit("commit-profile.store_unavailable", batch);
    expect(writeLog().length).toBe(before + 1);
  });
  expect((await memory.read())[0]).toBe("# Ada\nNew.\n");
});

test("test_learner_errors_chain_a_cause_without_the_key", async () => {
  const key = "sk-sentinel-profile-zqxw";
  const echo = `Incorrect API key provided: ${key.slice(0, 8)}****${key.slice(-4)} (${key})`;
  const refuse = (): never => {
    throw new Error(echo);
  };
  for (const [code, createModel] of [
    ["profile.provider", installModel([new Error(echo)]).createModel],
    ["profile.model_construction", refuse],
  ] as const) {
    const memory = await newMemory("# Ada\n", { llmConfig: { ...CONFIG, api_key: key }, createModel });
    const error = await withRun(memory, (session) => failure(() => session.learn(learnContext([], feedbackOf("x")))));
    const cause = (error.cause as Error).message;
    expect(error.code).toBe(code);
    expect(cause).toBe("Incorrect API key provided: [REDACTED] ([REDACTED])");
    expect(cause).not.toContain(key.slice(0, 8));
    expect(cause).not.toContain(key.slice(-4));
  }
});

test("test_learner_errors_scrub_the_key_variable_and_saved_login", async () => {
  for (const [key, login] of [
    ["sk-variable-profile-hgtw", false],
    ["sk-login-profile-bmrc", true],
  ] as const) {
    const vars = new Map([["YIAN_AUTH_FILE", join(home, `auth-${login}.json`)]]);
    const env: Env = (name) => vars.get(name);
    if (login) await new AuthStore(null, { env }).save("vllm", new ApiKeyCredentials(key), "profiler");
    else vars.set("VLLM_API_KEY", key);
    const echo = `Incorrect API key provided: ${key.slice(0, 8)}****${key.slice(-4)} (${key})`;
    const refuse = (): never => {
      throw new Error(echo);
    };
    for (const [code, createModel] of [
      ["profile.provider", installModel([new Error(echo)]).createModel],
      ["profile.model_construction", refuse],
    ] as const) {
      const memory = await newMemory("# Ada\n", {
        env,
        llmConfig: { provider: "vllm", model: "profiler" },
        createModel,
      });
      const error = await withRun(memory, (session) =>
        failure(() => session.learn(learnContext([], feedbackOf("x")))),
      );
      const cause = (error.cause as Error).message;
      expect(error.code).toBe(code);
      expect(cause).toBe("Incorrect API key provided: [REDACTED] ([REDACTED])");
      expect(cause).not.toContain(key.slice(0, 8));
      expect(cause).not.toContain(key.slice(-4));
    }
  }
});

test("test_session_recalls_and_learns_the_whole_document", async () => {
  const model = installModel(["<profile>\n# Ada\nBrief.\n</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  await withRun(memory, async (session) => {
    expect("openFrame" in session).toBe(false);
    expect(session.recall()).toBe("# Ada\n");
    expect(session.recall(RECALL)).toBe("# Ada\n");
    const mutations = await session.learn(learnContext([], feedbackOf("Be brief.")));
    expect(mutations.length).toBe(1);
    const mutation = mutations[0]!;
    expect("frameRef" in mutation).toBe(false);
    await session.commit("commit-1", [mutation]);
    const error = await failure(() => session.commit("commit-2", [mutation, mutation]));
    expect(error.code).toBe("profile.invalid_batch");
  });
  expect((await memory.read())[0]).toBe("# Ada\nBrief.\n");
  mkdirSync(join(home, "large"));
  const store = new FileMarkdownStore(join(home, "large", "ada.md"));
  await store.write(`# Ada\n${"x".repeat(40_000)}`, { baseVersion: (await store.read())[1] });
  const large = new ProfileMemory(store, { llmConfig: CONFIG, env: baseEnv(), maxChars: 100 });
  await withRun(large, async (session) => {
    // max_chars bounds learner replies, never recall.
    expect(session.recall(RECALL)).toBe((await large.read())[0]);
    expect(session.recall()).toBe((await large.read())[0]);
  });
});

/** An array subclass, which is not a plain mutation array. */
class Batch extends Array {}

test("commit's batch messages", async () => {
  const memory = await newMemory();
  await withRun(memory, async (session) => {
    const version = (await memory.read())[1];
    const mutation = documentMutation(version, "# Ada\nBrief.\n");
    const rows: [string, Mutable, string][] = [
      ["", [], "commit needs an id and a mutation array"],
      ["commit-1", mutation, "commit needs an id and a mutation array"],
      ["commit-1", [{ ...mutation }], "commit needs an id and a mutation array"],
      ["commit-1", Batch.from([mutation]), "commit needs an id and a mutation array"],
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
    const reason = (results[0] as PromiseRejectedResult).reason;
    expect(reason).toBeInstanceOf(MemoryOperationError);
    expect([reason.code, reason.retryable]).toEqual(["profile.version_mismatch", true]);
  });
  expect((await memory.read())[0]).toBe("# Ada\nBoth.\n");
  const line = writeLog().at(-1)!;
  expect(line).toContain('"feedback": [2]');
  expect(JSON.parse(line).provenance.feedback).toEqual([2]);
});

test("a store read failing with a system error during learn is profile.store_unavailable", async () => {
  const inner = new InMemoryMarkdownStore("# Ada\n");
  const down = systemError("down");
  let fail = false;
  const store = {
    read: (version?: string | null) => {
      if (fail) throw down;
      return inner.read(version);
    },
    write: (text: string, options: { baseVersion: string }) => inner.write(text, options),
  };
  const model = installModel(["<profile># Ada\nBrief.</profile>"]);
  const memory = new ProfileMemory(store, { llmConfig: CONFIG, env: baseEnv(), createModel: model.createModel });
  const error = await withRun(memory, (session) => {
    fail = true;
    return failure(() => session.learn(learnContext([], feedbackOf("x"))));
  });
  expect([error.stage, error.code, error.retryable, error.message]).toEqual([
    "learn",
    "profile.store_unavailable",
    true,
    "profile store could not be read",
  ]);
  expect(error.cause).toBe(down);
  expect(model.configs).toEqual([]);
});

test("feedback holding Infinity is profile.invalid_context after the model call", async () => {
  const model = installModel(["<profile># Ada\nBrief.</profile>"]);
  const memory = await newMemory("# Ada\n", { createModel: model.createModel });
  const error = await withRun(memory, (session) =>
    failure(() => session.learn(learnContext([], feedbackOf({ score: Number.POSITIVE_INFINITY }, 2n)))),
  );
  expect([error.stage, error.code, error.retryable, error.message]).toEqual([
    "learn",
    "profile.invalid_context",
    false,
    "feedback must be finite JSON",
  ]);
  expect(error.cause).toBeInstanceOf(RangeError);
  expect(model.requests.length).toBe(1);
});

test("onRetry sees a retried provider error", async () => {
  const retried = new ProviderError("rate_limit", "slow down", { retryAfter: 0.001 });
  const retries: RetryInfo[] = [];
  const memory = await newMemory("# Ada\n", {
    createModel: () => new ScriptedModel([retried, textReply("<profile># Ada\nBrief.</profile>")], CONFIG),
    onRetry: (retry) => {
      retries.push(retry);
    },
  });
  const batch = await withRun(memory, (session) => session.learn(learnContext([], feedbackOf("Be brief."))));
  expect(batch.length).toBe(1);
  expect(retries.map((retry) => [retry.attempt, retry.error])).toEqual([[1, retried]]);
});

/** The `ts/` directory. */
const ROOT = join(import.meta.dir, "../..");
const BUILT = join(ROOT, "dist/src/memory/profile.js");

/** This process's environment without `YIAN_*` keys, so a child reads no ambient yian settings. */
function childEnv(): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("YIAN_")));
}

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/memory_storage.test.ts`'s: importing a test file would run its tests.
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

describe.skipIf(!NODE)("the built backend on Node", () => {
  test(
    "Node opens a document Bun seeded, learns, commits, and Bun reads it back",
    async () => {
      const file = join(home, "ada.md");
      const seeded = await newMemory("# Ada\n");
      const seen = (await seeded.read())[1];
      const load = (name: string) => JSON.stringify(pathToFileURL(join(ROOT, `dist/src/${name}.js`)).href);
      const script = `
const assert = (await import("node:assert/strict")).default;
const { ProfileMemory } = await import(${load("memory/profile")});
const { FileMarkdownStore } = await import(${load("memory/storage/markdown")});
const c = await import(${load("memory/contracts")});
const { Message, Response, TextPart } = await import(${load("llm/messages")});
const { ScriptedModel } = await import(${load("llm/testing")});
const vars = new Map([["YIAN_AUTH_FILE", ${JSON.stringify(join(home, "auth.json"))}]]);
const createModel = (config) =>
  new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [new TextPart("<profile># Ada\\nCite sources.</profile>")] }) })],
    { provider: String(config.provider), model: String(config.model) },
  );
const memory = new ProfileMemory(new FileMarkdownStore(${JSON.stringify(file)}), {
  llmConfig: ${JSON.stringify(CONFIG)},
  env: (name) => vars.get(name),
  createModel,
});
try {
  const session = await memory.openRun();
  try {
    assert.equal(session.recall(), "# Ada\\n");
    const feedback = [{ channel: "output", target_name: null, value: "Cite sources.", owner_static_path: null, execution_seq: null }];
    const batch = await session.learn(new c.LearnContext("run-node", "profile", [], feedback));
    assert.equal(batch.length, 1);
    await session.commit("commit-node", batch);
  } finally {
    session.close();
  }
} finally {
  memory.close();
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
      const [text, version] = await new FileMarkdownStore(file).read();
      expect(text).toBe("# Ada\nCite sources.\n");
      const line = JSON.parse(writeLog().at(-1)!);
      expect([line.base_version, line.version]).toEqual([seen, version]);
      expect(line.provenance).toEqual({
        commit_id: "commit-node",
        run_id: "run-node",
        seen_version: seen,
        feedback: ["Cite sources."],
      });
    },
    30_000,
  );
});
