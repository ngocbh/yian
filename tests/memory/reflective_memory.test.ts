/**
 * Process-local reflective memory: snapshots and restore, sessions, recall, learning through a
 * scripted model, and commits. `baseEnv()` is each backend's environment, and `installModel(replies)`
 * makes the backend's `createModel` answer with one scripted reply per learn.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import crypto, { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BUILTIN, current, type Env, install } from "../../src/defaults.ts";
import { NoAuth } from "../../src/llm/auth.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import type { ChatModel, EventStream } from "../../src/llm/events.ts";
import { type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { Message, RefusalPart, type Request, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { MissingToolCallError, StructuredOutputError } from "../../src/llm/structured.ts";
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
  MemorySection,
  ModelTextInteraction,
  ModelValueInteraction,
  RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/contracts.ts";
import {
  boundedJsonMaterial,
  boundedJsonText,
  REFLECTION_SCHEMA,
  ReflectiveMemory,
  type ReflectiveSession,
  visibleInteractionJson,
} from "../../src/memory/reflective.ts";
import { logJson } from "../../src/typesys/json.ts";
import { type Value, writeAsJson } from "../../src/typesys/values.ts";

// Cloned snapshots and nested test values are edited freely.
type Mutable = any;

let home: string;

beforeEach(() => {
  install(BUILTIN);
  home = mkdtempSync(join(tmpdir(), "yian-reflective-"));
});

afterEach(() => {
  install(BUILTIN);
  rmSync(home, { recursive: true, force: true });
});

/** The test environment: an empty private store and an explicit model default. */
function baseEnv(overrides: Record<string, string> = {}): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
    ...Object.entries(overrides),
  ]);
  return (name) => vars.get(name);
}

/** A reflective memory reading `baseEnv()` unless `options.env` is given. */
function newMemory(options: ConstructorParameters<typeof ReflectiveMemory>[0] = {}): ReflectiveMemory {
  return new ReflectiveMemory({ env: baseEnv(), ...options });
}

/** The pinned snapshot of a fresh run of `memory`, optionally restored from `snapshot`. */
function snapshotOf(memory: ReflectiveMemory, snapshot?: JsonObject): JsonObject {
  const session = memory.openRun({ snapshot });
  try {
    return session.snapshot;
  } finally {
    session.close();
  }
}

/** Open a run, await `fn` with it, and close it. */
async function withRun<T>(
  memory: ReflectiveMemory,
  fn: (session: ReflectiveSession) => T | Promise<T>,
  snapshot?: JsonObject,
): Promise<T> {
  const session = memory.openRun({ snapshot });
  try {
    return await fn(session);
  } finally {
    session.close();
  }
}

/** A hand-built reflective mutation attributed to `run-1`. */
function mutation(mutationId: string, action: string, lessonId: string, text?: string): MemoryMutation {
  const payload: Record<string, unknown> = { action, lesson_id: lessonId, source: { run_id: "run-1" } };
  if (text !== undefined) payload.text = text;
  return new MemoryMutation(mutationId, payload);
}

/** Return the `MemoryOperationError` `fn` throws. */
function failure(fn: () => unknown): MemoryOperationError {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

/** Return the `MemoryOperationError` `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<MemoryOperationError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

const OLLAMA = { provider: "ollama", model: "qwen" };

const SYSTEM_TEXT =
  "You maintain atomic workflow lessons. Return only the requested structured actions. Use add for a " +
  "new id, revise/remove only for an existing id, or one noop when nothing should change. Never use tools.";

type Action = Record<string, unknown>;

const NOOP: Action[] = [{ action: "noop", lesson_id: null, text: null }];

/** A scripted assistant reply whose text is `text`. */
function textReply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });
}

/** A scripted reflector reply whose text is the JSON of `{actions}`. */
function reply(actions: readonly Action[]): Response {
  return textReply(JSON.stringify({ actions }));
}

/** A `RouteModel` on OpenAI Chat (`gpt-5.5`, so the forced tool strategy) that replays `replies`. */
class ToolRoute extends RouteModel {
  readonly script: ScriptedModel;

  constructor(replies: Response[]) {
    super({
      route: new Route({ provider: "openai", baseUrl: "https://api.example", path: "x", auth: new NoAuth() }),
      protocol: new OpenAIChat({ model: "gpt-5.5", provider: "openai" }),
    });
    this.script = new ScriptedModel(replies, { provider: "openai", model: "gpt-5.5" });
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    return this.script.stream(request, options);
  }
}

/**
 * A `createModel` option that records each config and returns a new
 * `ScriptedModel` (named by the config) answering with the next reply of the shared list.
 */
function installModel(replies: readonly (Response | Error)[]) {
  const queue = [...replies];
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

/** The user prompt of one reflector request. */
function promptOf(request: Request): string {
  return request.messages.at(-1)!.text();
}

/** The number of code points in `text`. */
function codePoints(text: string): number {
  return Array.from(text).length;
}

/** One succeeded episode of `agent` echoing `text`. */
function episode(runId = "run-1", text = "hello"): MemoryEpisode {
  return new MemoryEpisode(runId, [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { text }, {
    output: { answer: text.toUpperCase() },
  });
}

/** One learning context holding `episode(runId, text)`. */
function context(options: { runId?: string; text?: string; memory?: string } = {}): LearnContext {
  const { runId = "run-1", text = "hello", memory = "reflective" } = options;
  return new LearnContext(runId, memory, [episode(runId, text)]);
}

/** The most code points a reflection prompt can hold: three budgets and labels. */
function promptBudget(memory = "reflective"): number {
  const values = current();
  const labels = `Memory: ${memory}\nLimits: \nCurrent lessons: \nEpisodes: `;
  return (
    values.reflectiveMaxLessonChars +
    values.reflectiveMaxEpisodeChars +
    values.reflectiveMaxLimitChars +
    codePoints(labels)
  );
}

/** Learn `context()` in a fresh run of `memory`. */
function learnOnce(memory: ReflectiveMemory): Promise<MemoryMutation[]> {
  return withRun(memory, (session) => session.learn(context()));
}

test("constructor and protocol are exact", () => {
  for (const value of [true, 1.5, "1"]) {
    expect(() => newMemory({ maxItems: value as never })).toThrow(new TypeError("max_items must be an integer"));
    expect(() => newMemory({ recallLimit: value as never })).toThrow(new TypeError("recall_limit must be an integer"));
  }
  expect(() => newMemory({ recallLimit: 0 })).toThrow(new RangeError("recall_limit must be positive"));
  expect(() => newMemory({ maxItems: 0n })).toThrow(new RangeError("max_items must be positive"));
  expect(() => newMemory({ maxItems: 2n ** 53n })).toThrow(
    new RangeError("max_items must be at most 9007199254740991, got 9007199254740992"),
  );
  expect(() => newMemory({ llmConfig: { provider: "not-a-provider" } })).toThrow(RangeError);
  expect(() => newMemory({ llmConfig: { model: "" } })).toThrow(new RangeError("llm_config.model must be non-empty"));
  expect(() => newMemory({ llmConfig: { model: "  " } })).toThrow(new RangeError("llm_config.model must be non-empty"));
  for (const privateKey of ["api_key", "backend_token", "tenant_id"]) {
    expect(() => newMemory({ llmConfig: { ...OLLAMA, [privateKey]: "private" } })).toThrow(RangeError);
  }
  expect(() => newMemory({ llmConfig: { ...OLLAMA, api_key: "private" } })).toThrow(
    new RangeError(
      "llm_config.api_key is not supported by reflective memory, which records its model configuration",
    ),
  );
  for (const field of ["organization", "project", "headers"]) {
    expect(() => newMemory({ llmConfig: { provider: "openai", model: "m", [field]: "x" } })).toThrow(
      new RangeError(
        `llm_config.${field} is not supported by reflective memory, which records its model configuration`,
      ),
    );
  }
  expect(() => newMemory({ llmConfig: "config" as never })).toThrow(TypeError);
  expect(() => newMemory({ createParent: 1 as never })).toThrow(new TypeError("create_parent must be a bool"));

  const memory = newMemory({ maxItems: 7, recallLimit: 3 });
  expect(isMemory(memory)).toBe(true);
  expect(snapshotOf(memory).config).toEqual({ max_items: 7n, recall_limit: 3n });
  expect(snapshotOf(newMemory({ maxItems: 1.0, recallLimit: 2n })).config).toEqual({ max_items: 1n, recall_limit: 2n });
  expect(snapshotOf(newMemory({ maxItems: null, recallLimit: null })).config).toEqual({
    max_items: 100n,
    recall_limit: 20n,
  });
});

test("snapshot contains only public pinned model configuration", () => {
  const memory = newMemory({ llmConfig: { ...OLLAMA, ollama: { options: { temperature: 0.2 } } } });
  const llmConfig = snapshotOf(memory).llm_config as JsonObject;
  expect(Object.keys(llmConfig).toSorted()).toEqual(["model", "ollama", "provider"]);
  for (const key of ["api_key", "backend_token", "tenant_id"]) expect(Object.hasOwn(llmConfig, key)).toBe(false);
});

test("a number int in a host llm_config block is host intake", () => {
  const memory = newMemory({ llmConfig: { ...OLLAMA, ollama: { options: { num_ctx: 4096 } } } });
  const snapshot = snapshotOf(memory);
  expect(snapshot.llm_config).toEqual({ ...OLLAMA, ollama: { options: { num_ctx: 4096n } } });
  expect(snapshotOf(newMemory(), snapshot)).toEqual(snapshot);
});

test("defaults are read once and recorded snapshot never rereads them", () => {
  let reads = 0;
  const inner = baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: "qwen" });
  const env: Env = (name) => {
    if (name === "YIAN_DEFAULT_PROVIDER") reads += 1;
    return inner(name);
  };
  const first = newMemory({ env, llmConfig: { ollama: { options: { temperature: 0.2 } } } });
  const snapshot = snapshotOf(first);
  expect(reads).toBe(1);
  expect(snapshot.llm_config).toEqual({ ...OLLAMA, ollama: { options: { temperature: 0.2 } } });

  reads = 0;
  expect(snapshotOf(newMemory({ env }), snapshot)).toEqual(snapshot);
  expect(reads).toBe(0);
});

test("empty ambient model is rejected and open session config is pinned", async () => {
  const emptyEnv = baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: "" });
  const empty = failure(() => newMemory({ env: emptyEnv }).openRun());
  expect(empty.code).toBe("reflective.config");
  expect(empty.retryable).toBe(false);

  const defaultFailure = new Error("default-resolution-secret");
  const inner = baseEnv();
  const failDefault: Env = (name) => {
    if (name === "YIAN_DEFAULT_PROVIDER") throw defaultFailure;
    return inner(name);
  };
  const defaultError = failure(() => newMemory({ env: failDefault }).openRun());
  expect(defaultError.code).toBe("reflective.config");
  expect(defaultError.message).not.toContain("default-resolution-secret");
  expect(defaultError.cause).toBe(defaultFailure);

  const secretEnv = baseEnv({ YIAN_DEFAULT_PROVIDER: "secret-provider-value", YIAN_DEFAULT_MODEL: "qwen" });
  const configError = failure(() => newMemory({ env: secretEnv }).openRun());
  expect(configError.code).toBe("reflective.config");
  expect(configError.message).not.toContain("secret-provider-value");
  expect(configError.cause).toBeDefined();

  let pinnedAlready = false;
  const ollamaEnv = baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: "qwen" });
  const rereadEnv: Env = (name) => {
    if (pinnedAlready && name === "YIAN_DEFAULT_PROVIDER") throw new Error("defaults were reread");
    return ollamaEnv(name);
  };
  const model = installModel([reply(NOOP)]);
  const memory = newMemory({ env: rereadEnv, createModel: model.createModel });
  const session = memory.openRun();
  const pinned = session.snapshot.llm_config;
  expect(pinned).toEqual(OLLAMA);
  pinnedAlready = true;
  expect(await session.learn(context())).toEqual([]);
  expect(model.configs as unknown[]).toEqual([pinned]);
  session.close();
});

test("snapshot is defensive and foreign restore is conservative", () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const session = memory.openRun();
  const snapshot = session.snapshot;
  expect(() => {
    (snapshot as Mutable).namespace = "changed";
  }).toThrow(TypeError);
  const recorded = session.snapshot;
  expect(recorded.namespace).not.toBe("changed");
  session.close();

  const foreign = newMemory();
  snapshotOf(foreign);
  const foreignError = failure(() => foreign.openRun({ snapshot: recorded }));
  expect(foreignError.code).toBe("reflective.invalid_snapshot");
  expect(foreignError.message).toContain("namespace");

  const reset = newMemory();
  reset.reset();
  expect(failure(() => reset.openRun({ snapshot: recorded })).message).toContain("namespace");
});

test("pristine restore keeps recorded limits", async () => {
  const snapshot = snapshotOf(newMemory({ llmConfig: OLLAMA, maxItems: 4, recallLimit: 2 }));
  await withRun(
    newMemory(),
    async (session) => {
      expect(await session.recall()).toBe("");
      expect(session.snapshot).toEqual(snapshot);
    },
    snapshot,
  );
});

test("unset limits adopt recorded ones and explicit ones must match", () => {
  expect(snapshotOf(newMemory({ llmConfig: OLLAMA })).config).toEqual({ max_items: 100n, recall_limit: 20n });
  const recorded = snapshotOf(newMemory({ llmConfig: OLLAMA, maxItems: 4, recallLimit: 2 }));
  expect(snapshotOf(newMemory(), recorded)).toEqual(recorded);
  expect(snapshotOf(newMemory({ maxItems: 4 }), recorded)).toEqual(recorded);
  for (const options of [{ maxItems: 5 }, { recallLimit: 3 }]) {
    const error = failure(() => newMemory(options).openRun({ snapshot: recorded }));
    expect(error.code).toBe("reflective.config_conflict");
    expect(error.message).toBe("configured limits differ from the recorded limits");
    expect(error.retryable).toBe(false);
  }
});

test("snapshot rejects noncanonical scalars and circular json", () => {
  const circular: Mutable = {};
  circular.value = circular;
  expect(() => newMemory({ llmConfig: circular })).toThrow(
    new RangeError("llm_config.value contains a circular JSON value"),
  );

  const memory = newMemory({ llmConfig: OLLAMA });
  const version: Mutable = structuredClone(snapshotOf(memory));
  version.version = true;
  expect(failure(() => newMemory().openRun({ snapshot: version })).code).toBe("reflective.invalid_snapshot");

  const blankModel: Mutable = structuredClone(snapshotOf(memory));
  blankModel.llm_config.model = "   ";
  const error = failure(() => newMemory().openRun({ snapshot: blankModel }));
  expect(error.code).toBe("reflective.invalid_snapshot");
  expect(error.message).toBe("recorded memory snapshot is invalid");
  expect((error.cause as Error).message).toBe("snapshot.llm_config must pin nonblank provider and model");

  for (const [field, value] of [
    ["revision", 2n ** 60n],
    ["epoch", -1n],
    ["version", 1n],
  ] as const) {
    const changed: Mutable = structuredClone(snapshotOf(memory));
    changed[field] = value;
    const changedError = failure(() => newMemory().openRun({ snapshot: changed }));
    expect(changedError.code).toBe("reflective.invalid_snapshot");
    if (field === "revision") {
      expect((changedError.cause as Error).message).toBe("snapshot.revision must be at most 9007199254740991");
    }
  }
});

test("reflective json rejects non utf8 model and snapshot strings", () => {
  const surrogate = "\ud800";
  expect(() => newMemory({ llmConfig: { provider: "ollama", model: surrogate } })).toThrow(
    new RangeError("Unicode surrogate is not valid canonical JSON"),
  );
  const snapshot: Mutable = structuredClone(snapshotOf(newMemory({ llmConfig: OLLAMA })));
  snapshot.namespace = surrogate;
  expect(failure(() => newMemory().openRun({ snapshot })).code).toBe("reflective.invalid_snapshot");
});

test("snapshot rejects excessive nesting as typed invalid snapshot", () => {
  const snapshot: Mutable = structuredClone(snapshotOf(newMemory({ llmConfig: OLLAMA })));
  let nested: Mutable = {};
  for (let i = 0; i < 1_500; i++) nested = { nested };
  snapshot.extra = nested;
  const error = failure(() => newMemory().openRun({ snapshot }));
  expect(error.code).toBe("reflective.invalid_snapshot");
  expect((error.cause as Error).message).toBe("snapshot has invalid fields");
});

for (const [field, value] of [
  ["sandbox", "read-only"],
  ["cwd", "/tmp"],
  ["provider", "codex"],
] as const) {
  test(`old reflective snapshot rejects active retired options: ${field}`, () => {
    const memory = newMemory({ llmConfig: { provider: "openai", model: "old-model" } });
    const old: Mutable = structuredClone(snapshotOf(memory));
    delete old.llm_config._connection;
    Object.assign(old.llm_config, { sandbox: null, cwd: null });
    old.llm_config[field] = value;
    expect(failure(() => newMemory().openRun({ snapshot: old })).code).toBe("reflective.invalid_snapshot");
  });
}

test("same object restore accepts recorded snapshot", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "one", "first")]));
  const snapshot = snapshotOf(memory);
  await withRun(
    memory,
    async (restored) => {
      expect(await restored.recall()).toBe("[one] first");
      expect(restored.snapshot).toEqual(snapshot);
    },
    snapshot,
  );
});

test("populated memory rejects foreign snapshot", async () => {
  const snapshot = snapshotOf(newMemory({ llmConfig: OLLAMA }));
  const populated = newMemory({ llmConfig: OLLAMA });
  await withRun(populated, (session) => session.commit("populate", [mutation("populate", "add", "one", "first")]));
  expect(failure(() => populated.openRun({ snapshot })).message).toContain("namespace");
});

test("snapshot rejects malformed foreign and extra lesson sources", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "one", "first")]));
  const base = snapshotOf(memory);
  const validSource = { run_id: "run-1" };
  for (const malformed of [
    "not-an-object",
    {},
    { run_id: "" },
    { ...validSource, frame_ref: {} },
    { ...validSource, extra: "source-secret" },
  ]) {
    const snapshot: Mutable = structuredClone(base);
    snapshot.lessons[0].source = malformed;
    const error = failure(() => newMemory().openRun({ snapshot }));
    expect(error.code).toBe("reflective.invalid_snapshot");
    expect(error.message).not.toContain("source-secret");
  }
});

test("session idempotency and closed session operations", async () => {
  const memory = newMemory();
  const session = memory.openRun();
  session.close();
  session.close();
  expect((await rejection(session.recall())).message).toContain("closed");
  expect((await rejection(session.learn(context()))).message).toContain("closed");
  expect(failure(() => session.commit("commit", [])).message).toContain("closed");
  // The failed recall and learn released their operations, so the backend isn't busy.
  memory.close();
});

test("sessions of one instance share one lesson list and limit", async () => {
  const memory = newMemory({ llmConfig: OLLAMA, maxItems: 2, recallLimit: 1 });
  await withRun(memory, (session) =>
    session.commit("seed", [
      mutation("one", "add", "one", "first lesson"),
      mutation("two", "add", "two", "second lesson"),
    ]),
  );
  await withRun(memory, async (session) => {
    const snapshot = session.snapshot;
    expect(snapshot.version).toBe(2n);
    expect(Object.keys(snapshot).toSorted()).toEqual([
      "config",
      "epoch",
      "format",
      "lessons",
      "llm_config",
      "namespace",
      "revision",
      "version",
    ]);
    expect(snapshot.config).toEqual({ max_items: 2n, recall_limit: 1n });
    expect((snapshot.lessons as JsonObject[]).map((lesson) => lesson.source)).toEqual([
      { run_id: "run-1" },
      { run_id: "run-1" },
    ]);
    expect(await session.recall()).toBe("[one] first lesson\n[two] second lesson");
    const error = failure(() => session.commit("overflow", [mutation("three", "add", "three", "third")]));
    expect(error.code).toBe("reflective.max_items");
  });
});

test("recall all and context ranking are deterministic", async () => {
  const model = installModel([
    reply([
      { action: "add", lesson_id: "b", text: "red apple" },
      { action: "add", lesson_id: "a", text: "blue berry" },
      { action: "add", lesson_id: "c", text: "red berry" },
    ]),
  ]);
  const memory = newMemory({ llmConfig: OLLAMA, recallLimit: 2, createModel: model.createModel });
  const mutations = await learnOnce(memory);
  await withRun(memory, (session) => session.commit("seed", mutations));
  await withRun(memory, async (session) => {
    expect(await session.recall()).toBe("[b] red apple\n[a] blue berry\n[c] red berry");
    const recalled = await session.recall(new RecallContext("", "RED berry", { tone: "BLUE" }, ["root", "user"]));
    expect(recalled).toBe("[a] blue berry\n[c] red berry");
  });
});

test("subclass controls query rewriting", async () => {
  // The async override is awaited.
  class AppleMemory extends ReflectiveMemory {
    override rewriteQuery(): string {
      return "apple";
    }
  }
  class AsyncAppleMemory extends ReflectiveMemory {
    override async rewriteQuery(): Promise<string> {
      await Promise.resolve();
      return "apple";
    }
  }
  for (const Memory of [AppleMemory, AsyncAppleMemory]) {
    const model = installModel([
      reply([
        { action: "add", lesson_id: "berry", text: "fresh berry" },
        { action: "add", lesson_id: "apple", text: "green apple" },
      ]),
    ]);
    const memory = new Memory({ env: baseEnv(), llmConfig: OLLAMA, recallLimit: 1, createModel: model.createModel });
    const mutations = await learnOnce(memory);
    await withRun(memory, (session) => session.commit("seed", mutations));
    await withRun(memory, async (session) => {
      expect(await session.recall(new RecallContext("", "berry", {}, ["root"]))).toBe("[apple] green apple");
    });
  }
});

test("query rewriting failure is sanitized", async () => {
  const cause = new Error("query-secret");
  class FailingMemory extends ReflectiveMemory {
    override rewriteQuery(): string {
      throw cause;
    }
  }
  const memory = new FailingMemory({ env: baseEnv(), llmConfig: OLLAMA });
  const error = await withRun(memory, (session) =>
    rejection(session.recall(new RecallContext("", "hello", {}, ["root"]))),
  );
  expect(error.code).toBe("reflective.query");
  expect(error.message).not.toContain("query-secret");
  expect(error.cause).toBe(cause);
});

test("recall rejects an invalid context and a non-string query", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const context = await withRun(memory, (session) => rejection(session.recall({} as never)));
  expect([context.stage, context.code, context.message]).toEqual([
    "recall",
    "reflective.invalid_context",
    "context must be a RecallContext or null",
  ]);

  class NumberMemory extends ReflectiveMemory {
    override rewriteQuery(): string {
      return 1 as never;
    }
  }
  const numeric = new NumberMemory({ env: baseEnv(), llmConfig: OLLAMA });
  const query = await withRun(numeric, (session) =>
    rejection(session.recall(new RecallContext("", "hello", {}, ["root"]))),
  );
  expect([query.stage, query.code, query.message]).toEqual([
    "recall",
    "reflective.query",
    "rewriteQuery() must return a string",
  ]);
  // The failed recalls released their operations, so the backends aren't busy.
  memory.close();
  numeric.close();
});

test("commit is atomic idempotent and empty batches advance revision", async () => {
  const model = installModel([reply([{ action: "add", lesson_id: "one", text: "first" }])]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const add = await learnOnce(memory);
  await withRun(memory, (session) => {
    session.commit("add", add);
    session.commit("add", add);
    expect(failure(() => session.commit("add", [])).message).toContain("collision");
  });
  const before = snapshotOf(memory).revision as bigint;
  await withRun(memory, (session) => session.commit("empty", []));
  expect(snapshotOf(memory).revision).toBe(before + 1n);
});

test("revise and remove are learned and committed", async () => {
  const model = installModel([
    reply([{ action: "add", lesson_id: "one", text: "first" }]),
    reply([{ action: "revise", lesson_id: "one", text: "revised" }]),
    reply([{ action: "remove", lesson_id: "one", text: null }]),
  ]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const add = await learnOnce(memory);
  await withRun(memory, (session) => session.commit("add", add));
  const revise = await learnOnce(memory);
  await withRun(memory, (session) => session.commit("revise", revise));
  await withRun(memory, async (session) => expect(await session.recall()).toBe("[one] revised"));
  const remove = await learnOnce(memory);
  await withRun(memory, (session) => session.commit("remove", remove));
  await withRun(memory, async (session) => expect(await session.recall()).toBe(""));
});

test("learn and commit enforce final max items", async () => {
  const model = installModel([
    reply([
      { action: "add", lesson_id: "new", text: "new lesson" },
      { action: "remove", lesson_id: "old", text: null },
    ]),
    reply([
      { action: "add", lesson_id: "two", text: "two" },
      { action: "add", lesson_id: "three", text: "three" },
    ]),
  ]);
  const memory = newMemory({ llmConfig: OLLAMA, maxItems: 1, createModel: model.createModel });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "old", "old lesson")]));
  await withRun(memory, async (session) => session.commit("replace", await session.learn(context())));
  await withRun(memory, async (session) => expect(await session.recall()).toBe("[new] new lesson"));

  const empty = newMemory({ llmConfig: OLLAMA, maxItems: 1, createModel: model.createModel });
  const learnError = await withRun(empty, (session) => rejection(session.learn(context())));
  expect([learnError.code, learnError.message]).toEqual(["reflective.max_items", "proposed lessons exceed max_items"]);
  const error = await withRun(empty, async (session) => {
    const commitError = failure(() =>
      session.commit("too-many", [mutation("two", "add", "two", "two"), mutation("three", "add", "three", "three")]),
    );
    expect(await session.recall()).toBe("");
    expect(session.snapshot.revision).toBe(0n);
    return commitError;
  });
  expect([error.code, error.message]).toEqual(["reflective.max_items", "mutation batch exceeds max_items"]);
});

test("multi mutation commit failure has no partial effect", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const error = await withRun(memory, async (session) => {
    const commitError = failure(() =>
      session.commit("atomic", [
        mutation("left", "add", "one", "first"),
        mutation("right", "revise", "missing", "bad"),
      ]),
    );
    expect(await session.recall()).toBe("");
    expect(session.snapshot.revision).toBe(0n);
    return commitError;
  });
  expect(error.code).toBe("reflective.invalid_mutation");
  await withRun(memory, async (session) => expect(await session.recall()).toBe(""));
});

test("concurrent same generation commits have one winner", async () => {
  const model = installModel([
    reply([{ action: "add", lesson_id: "one", text: "first" }]),
    reply([{ action: "add", lesson_id: "two", text: "second" }]),
  ]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const first = await learnOnce(memory);
  const second = await learnOnce(memory);
  const left = memory.openRun();
  const right = memory.openRun();
  left.commit("left", first);
  const error = failure(() => right.commit("right", second));
  expect([error.stage, error.code, error.message, error.retryable]).toEqual([
    "commit",
    "reflective.stale_generation",
    "memory session is stale",
    false,
  ]);
  left.close();
  right.close();
});

test("reset preserves commit ledger ordering and old recall", async () => {
  const model = installModel([reply([{ action: "add", lesson_id: "one", text: "first" }])]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const add = await learnOnce(memory);
  await withRun(memory, (seed) => seed.commit("applied", add));
  const old = memory.openRun();
  expect(await old.recall()).toBe("[one] first");
  memory.reset();

  expect(await old.recall()).toBe("[one] first");
  old.commit("applied", add);
  expect(failure(() => old.commit("applied", [])).message).toContain("collision");
  expect(failure(() => old.commit("not-applied", add)).message).toContain("stale");
  await withRun(memory, async (fresh) => expect(await fresh.recall()).toBe(""));
  old.close();
});

test("commit rejects invalid batches, commit ids and mutations with their messages", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "kept", "kept lesson")]));
  const source = { run_id: "run-1" };
  const rows: [string, unknown, unknown, string][] = [
    ["reflective.invalid_commit_id", "", [], "commit id must be a nonempty string"],
    ["reflective.invalid_commit_id", 1, [], "commit id must be a nonempty string"],
    ["reflective.invalid_batch", "c", {}, "mutations must be an exact array"],
    ["reflective.invalid_batch", "c", [{}], "mutation batch contains an invalid value"],
    [
      "reflective.invalid_batch",
      "c",
      [mutation("m", "add", "a", "x"), mutation("m", "add", "b", "y")],
      "mutation ids must be unique within a batch",
    ],
    ["reflective.invalid_mutation", "c", [new MemoryMutation("m", [])], "mutation payload must be an object"],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "a" })],
      "mutation payload has invalid fields",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "a", source, text: "x", extra: 1 })],
      "mutation payload has invalid fields",
    ],
    ["reflective.invalid_mutation", "c", [mutation("m", "merge", "a", "x")], "mutation action is invalid"],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "", source, text: "x" })],
      "mutation identity or source is invalid",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "a", source: { run_id: "r", extra: 1 }, text: "x" })],
      "mutation identity or source is invalid",
    ],
    ["reflective.invalid_mutation", "c", [mutation("m", "add", "a")], "mutation text is invalid"],
    ["reflective.invalid_mutation", "c", [mutation("m", "add", "a", "  ")], "text must not be blank"],
    ["reflective.invalid_mutation", "c", [mutation("m", "remove", "kept", "x")], "remove mutations cannot carry text"],
    [
      "reflective.invalid_mutation",
      "c",
      [mutation("m1", "add", "a", "x"), mutation("m2", "revise", "a", "y")],
      "a batch cannot target one lesson more than once",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [mutation("m", "add", "kept", "x")],
      "add mutation targets an existing lesson",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [mutation("m", "revise", "a", "x")],
      "revise mutation targets a missing lesson",
    ],
    ["reflective.invalid_mutation", "c", [mutation("m", "remove", "a")], "remove mutation targets a missing lesson"],
  ];
  await withRun(memory, async (session) => {
    for (const [code, commitId, mutations, message] of rows) {
      const error = failure(() => session.commit(commitId as never, mutations as never));
      expect([error.stage, error.code, error.message, error.retryable]).toEqual(["commit", code, message, false]);
    }
    expect(await session.recall()).toBe("[kept] kept lesson");
    expect(session.snapshot.revision).toBe(1n);
  });
});

test("a host mutation payload holding -0 cannot be encoded canonically", async () => {
  await withRun(newMemory({ llmConfig: OLLAMA }), (session) => {
    const zero = new MemoryMutation("m", { action: "add", lesson_id: "a", source: { run_id: "r" }, text: "x", n: -0 });
    const error = failure(() => session.commit("zero", [zero]));
    expect([error.code, error.message]).toEqual([
      "reflective.invalid_batch",
      "mutation batch cannot be encoded canonically",
    ]);
    expect(error.cause).toBeInstanceOf(TypeError);
  });
});

test("a re-entrant close from inside an async rewriteQuery is busy", async () => {
  let closeError: unknown = null;
  class ClosingQueryMemory extends ReflectiveMemory {
    override async rewriteQuery(): Promise<string> {
      await Promise.resolve();
      try {
        this.close();
      } catch (error) {
        closeError = error;
      }
      return "query";
    }
  }
  const memory = new ClosingQueryMemory({ env: baseEnv(), llmConfig: OLLAMA });
  await withRun(memory, async (session) => {
    expect(await session.recall(new RecallContext("", "", {}, ["root"]))).toBe("");
  });
  expect(closeError).toBeInstanceOf(MemoryOperationError);
  const busy = closeError as MemoryOperationError;
  expect([busy.stage, busy.code, busy.message, busy.retryable]).toEqual([
    "close",
    "reflective.busy",
    "memory backend is active",
    true,
  ]);
  memory.close();
  expect(failure(() => memory.openRun()).code).toBe("reflective.closed");
});

test("reflective episode material retains the physical execution sequence", async () => {
  const model = installModel([reply(NOOP)]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const sequenced = new MemoryEpisode("run-1", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { text: "hello" }, {
    output: "HELLO",
    executionSeq: 17,
  });
  await withRun(memory, (session) => session.learn(new LearnContext("run-1", "reflective", [sequenced])));
  const material = JSON.parse(promptOf(model.requests[0]!).split("Episodes: ")[1]!);
  expect(material.episodes[0].execution_seq).toBe(17);
  expect(material.episodes[0].owner_static_path).toEqual(["agent"]);
});

test("learn uses pinned config and returns deterministic proposals", async () => {
  const add = { action: "add", lesson_id: "one", text: "Keep it short" };
  const model = installModel([reply([add, ...NOOP]), reply([add]), reply([add])]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  let pinned: unknown;
  const noop = await withRun(memory, (session) => {
    pinned = session.snapshot.llm_config;
    return rejection(session.learn(context()));
  });
  expect(noop.message).toContain("no-op");
  expect(model.configs).toEqual([pinned as Record<string, unknown>]);
  const request = model.requests[0]!;
  expect(request.outputFormat?.name).toBe("_ReflectionResponse");
  expect(request.messages.map((message) => message.role)).toEqual(["system", "user"]);
  expect(request.messages[0]!.text()).toBe(SYSTEM_TEXT);

  await withRun(memory, async (session) => {
    const first = await session.learn(context());
    const second = await session.learn(context());
    expect(first).toEqual(second);
    const payload = { action: "add", lesson_id: "one", source: { run_id: "run-1" }, text: "Keep it short" };
    expect(first[0]!.payload).toEqual(payload);
    const identity = logJson({ index: 0n, memory: "reflective", payload, run_id: "run-1" });
    expect(first[0]!.mutationId).toBe(`reflective:${createHash("sha256").update(identity).digest("hex")}`);
  });
});

test("learn rejects malformed or conflicting actions", async () => {
  const rows: [Action[], string, string][] = [
    [[{ action: "add", lesson_id: "", text: "x" }], "reflective.contract", "lesson ids must be non-empty strings"],
    [
      [{ action: "add", lesson_id: "one", text: "   " }],
      "reflective.contract",
      "add and revise actions require non-blank text",
    ],
    [
      [{ action: "remove", lesson_id: "missing", text: null }],
      "reflective.contract",
      "reflector remove action targets a missing lesson",
    ],
    [
      [
        { action: "add", lesson_id: "same", text: "x" },
        { action: "add", lesson_id: "same", text: "y" },
      ],
      "reflective.contract",
      "actions cannot target one lesson more than once",
    ],
    // Learn's other action messages.
    [
      [{ action: "noop", lesson_id: "kept", text: null }],
      "reflective.contract",
      "a no-op action cannot carry an id or text",
    ],
    [[{ action: "add", text: "x" }], "reflective.contract", "lesson ids must be non-empty strings"],
    [[{ action: "remove", lesson_id: "kept", text: "x" }], "reflective.contract", "remove actions cannot carry text"],
    [
      [{ action: "add", lesson_id: "kept", text: "x" }],
      "reflective.contract",
      "reflector add action targets an existing lesson",
    ],
    [
      [{ action: "revise", lesson_id: "missing", text: "x" }],
      "reflective.contract",
      "reflector revise action targets a missing lesson",
    ],
  ];
  const model = installModel(rows.map(([actions]) => reply(actions)));
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "kept", "kept lesson")]));
  for (const [, code, message] of rows) {
    const error = await withRun(memory, (session) => rejection(session.learn(context())));
    expect([error.stage, error.code, error.message, error.retryable]).toEqual(["learn", code, message, false]);
  }
});

test("model construction failure is final and sanitized", async () => {
  const cause = new Error("construction-secret");
  const memory = newMemory({
    llmConfig: OLLAMA,
    createModel: () => {
      throw cause;
    },
  });
  const error = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([error.code, error.message, error.retryable]).toEqual([
    "reflective.model_construction",
    "reflector model construction failed",
    false,
  ]);
  expect(error.cause).toBe(cause);
});

test("structured binding failure is final and sanitized", async () => {
  const memory = newMemory({
    llmConfig: OLLAMA,
    createModel: () => new ScriptedModel([], { provider: "scripted", model: "no-structured-sentinel" }),
  });
  const error = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([error.code, error.message, error.retryable]).toEqual([
    "reflective.structured_output",
    "reflector does not support the required structured output",
    false,
  ]);
  expect(error.cause).toBeInstanceOf(TypeError);
  expect((error.cause as Error).message).toBe("the reflector model has no native structured output");
});

test("learn provider failure is retryable and sanitized", async () => {
  const cause = new Error("provider-secret");
  const model = installModel([cause]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const error = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([error.code, error.message, error.retryable]).toEqual([
    "reflective.provider",
    "reflector invocation failed",
    true,
  ]);
  expect(error.cause).toBe(cause);
});

test("onRetry sees a retried provider error", async () => {
  const retried = new ProviderError("rate_limit", "slow down", { retryAfter: 0.001 });
  const retries: RetryInfo[] = [];
  const memory = newMemory({
    llmConfig: OLLAMA,
    createModel: () => new ScriptedModel([retried, reply(NOOP)], OLLAMA),
    onRetry: (retry) => {
      retries.push(retry);
    },
  });
  expect(await learnOnce(memory)).toEqual([]);
  expect(retries.map((retry) => [retry.attempt, retry.error])).toEqual([[1, retried]]);
});

test("onRetry sees the retried error scrubbed of the call's credentials", async () => {
  const key = "sk-variable-reflect-retry";
  const env = baseEnv({ VLLM_API_KEY: key });
  const config = { provider: "vllm", model: "m" };
  const retried = new ProviderError("rate_limit", `Incorrect API key provided: ${key}`, { retryAfter: 0.001 });
  const retries: RetryInfo[] = [];
  const memory = new ReflectiveMemory({
    env,
    llmConfig: config,
    createModel: () => new ScriptedModel([retried, reply(NOOP)], config),
    onRetry: (retry) => {
      retries.push(retry);
    },
  });
  expect(await learnOnce(memory)).toEqual([]);
  expect(retries.map(({ attempt, error }) => [attempt, error.kind, error.message])).toEqual([
    [1, "rate_limit", "Incorrect API key provided: [REDACTED]"],
  ]);
});

test("learn errors scrub the key variable and saved login", async () => {
  for (const [key, login] of [
    ["sk-variable-reflect-hgtw", false],
    ["sk-login-reflect-bmrc", true],
  ] as const) {
    const vars = new Map([["YIAN_AUTH_FILE", join(home, `auth-${login}.json`)]]);
    const env: Env = (name) => vars.get(name);
    if (login) await new AuthStore(null, { env }).save("vllm", new ApiKeyCredentials(key), "m");
    else vars.set("VLLM_API_KEY", key);
    const echo = `Incorrect API key provided: ${key.slice(0, 8)}****${key.slice(-4)} (${key})`;
    const refuse = () => {
      throw new Error(echo);
    };
    for (const [code, createModel] of [
      ["reflective.model_construction", refuse],
      ["reflective.provider", installModel([new Error(echo)]).createModel],
    ] as const) {
      const memory = new ReflectiveMemory({ env, llmConfig: { provider: "vllm", model: "m" }, createModel });
      const error = await withRun(memory, (session) => rejection(session.learn(context())));
      const cause = (error.cause as Error).message;
      expect(error.code).toBe(code);
      expect(cause).toContain("[REDACTED]");
      expect(cause).not.toContain(key.slice(0, 8));
      expect(cause).not.toContain(key.slice(-4));
    }
  }
});

test("structured output parser failure is a final contract error", async () => {
  const model = installModel([textReply("parser-secret")]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const error = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([error.code, error.message, error.retryable]).toEqual([
    "reflective.contract",
    "reflector returned invalid structured output: the reply held no JSON object",
    false,
  ]);
  expect(error.cause).toBeInstanceOf(StructuredOutputError);
});

test("a refused reflector reply is a contract error that says so without the refusal's text", async () => {
  const refused = new Response({
    message: new Message({ role: "assistant", parts: [new RefusalPart("refusal-secret")] }),
  });
  const model = installModel([refused]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const error = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([error.code, error.message, error.retryable]).toEqual([
    "reflective.contract",
    "reflector returned invalid structured output: the model refused",
    false,
  ]);
  expect(error.cause).toBeInstanceOf(StructuredOutputError);
});

test("a tool-strategy reflector reply without a valid answer names the defect", async () => {
  const call = new ToolCallPart({ id: "c1", name: "_ReflectionResponse", arguments: { actions: [] } });
  const replies = [
    textReply('{"actions":[{"action":"noop"}]}'),
    new Response({ message: new Message({ role: "assistant", parts: [call] }) }),
  ];
  const models: ToolRoute[] = [];
  const createModel = () => {
    const model = new ToolRoute([replies[models.length]!]);
    models.push(model);
    return model;
  };
  const memory = newMemory({ llmConfig: { provider: "openai", model: "gpt-5.5" }, createModel });
  const missing = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([missing.code, missing.message, missing.retryable]).toEqual([
    "reflective.contract",
    "reflector returned invalid structured output: the reply did not call the _ReflectionResponse tool; call it with the answer",
    false,
  ]);
  expect(missing.cause).toBeInstanceOf(MissingToolCallError);
  expect(models[0]!.script.requests[0]!.tools.map((tool) => tool.name)).toEqual(["_ReflectionResponse"]);

  const empty = await withRun(memory, (session) => rejection(session.learn(context())));
  expect([empty.code, empty.message, empty.retryable]).toEqual([
    "reflective.contract",
    "reflector returned invalid structured output: actions must not be empty",
    false,
  ]);
  expect(empty.cause).toBeInstanceOf(TypeError);
});

test("parsed response validation failure is final and sanitized", async () => {
  const replies = [
    { actions: [{ action: "add", lesson_id: "one", text: "text-secret", unexpected: "response-secret" }] },
    // The other reply defects.
    {},
    { actions: [{ action: "noop" }], extra: "response-secret" },
    { actions: "response-secret" },
    { actions: [] },
    { actions: ["response-secret"] },
    { actions: [{}] },
    { actions: [{ action: "response-secret" }] },
    { actions: [{ action: "add", lesson_id: 1, text: "text-secret" }] },
    { actions: [{ action: "add", lesson_id: "one", text: ["text-secret"] }] },
  ];
  const defects = [
    "actions[0] may hold only action, lesson_id and text",
    "the reply must have exactly the key actions",
    "the reply must have exactly the key actions",
    "actions must be an array",
    "actions must not be empty",
    "actions[0] must be an object",
    "actions[0].action must be add, revise, remove or noop",
    "actions[0].action must be add, revise, remove or noop",
    "actions[0].lesson_id must be a string or null",
    "actions[0].text must be a string or null",
  ];
  const model = installModel(replies.map((value) => textReply(JSON.stringify(value))));
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  for (const defect of defects) {
    const error = await withRun(memory, (session) => rejection(session.learn(context())));
    expect([error.code, error.message, error.retryable]).toEqual([
      "reflective.contract",
      `reflector returned invalid structured output: ${defect}`,
      false,
    ]);
    expect(error.cause).toBeInstanceOf(TypeError);
    expect((error.cause as Error).message).toBe(defect);
  }

  // Absent `lesson_id` and `text` are `null`.
  const absent = installModel([reply([{ action: "noop" }])]);
  expect(await learnOnce(newMemory({ llmConfig: OLLAMA, createModel: absent.createModel }))).toEqual([]);
});

test("episode material is capped in one model call", async () => {
  const model = installModel([reply(NOOP)]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const huge = "x".repeat(current().reflectiveMaxEpisodeChars * 2);
  expect(await withRun(memory, (session) => session.learn(context({ text: huge })))).toEqual([]);
  expect(model.configs).toHaveLength(1);
  expect(model.requests).toHaveLength(1);
  const material = promptOf(model.requests[0]!);
  expect(codePoints(material)).toBeLessThanOrEqual(promptBudget());
  expect(material).toContain("truncated");
});

test("huge current lessons cannot displace new episode evidence", async () => {
  const model = installModel([reply(NOOP)]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  await withRun(memory, (session) =>
    session.commit("large-lesson", [
      mutation("large-lesson", "add", "large", `old-evidence-${"x".repeat(40_000)}`),
    ]),
  );
  expect(await withRun(memory, (session) => session.learn(context({ text: "new-evidence" })))).toEqual([]);
  const prompt = promptOf(model.requests[0]!);
  expect(codePoints(prompt)).toBeLessThanOrEqual(promptBudget());
  for (const part of ["Current lessons:", "truncated", "Episodes:", "new-evidence", "NEW-EVIDENCE"]) {
    expect(prompt).toContain(part);
  }
});

test("learning is propose only and rejects a stale generation", async () => {
  const add = reply([{ action: "add", lesson_id: "one", text: "first" }]);
  const model = installModel([add, add]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const session = memory.openRun();
  const proposals = await session.learn(context());
  expect(await session.recall()).toBe("");
  await withRun(memory, (winner) => winner.commit("winner", proposals));
  const error = await rejection(session.learn(context()));
  expect([error.stage, error.code, error.message]).toEqual([
    "learn",
    "reflective.stale_generation",
    "memory session is stale",
  ]);
  session.close();
});

test("reflective memory implements protocol and imports only lower layers", () => {
  expect(isMemory(newMemory())).toBe(true);
  const source = readFileSync(join(import.meta.dirname, "../../src/memory/reflective.ts"), "utf8");
  const specifiers = [...source.matchAll(/\bfrom\s+"([^"]+)"|\bimport\s*\(?\s*"([^"]+)"/g)].map(
    (match) => (match[1] ?? match[2])!,
  );
  expect(specifiers.length).toBeGreaterThan(0);
  const forbidden = ["nodes", "compile", "compose", "runtime", "run", "cli"];
  for (const specifier of specifiers) {
    const parts = specifier.replace(/\.[jt]s$/, "").split("/");
    expect(parts.filter((part) => forbidden.includes(part))).toEqual([]);
  }
  logJson(snapshotOf(newMemory()));
});

test("session reflects under the memory key", async () => {
  const cite = reply([{ action: "add", lesson_id: "a", text: "Cite" }]);
  const model = installModel([cite, cite]);
  const memory = newMemory({ createModel: model.createModel });
  const noted = new MemoryEpisode("run-1", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { text: "hi" }, {
    output: "HI",
    memorySections: [new MemorySection("notes", "prior")],
  });
  const [notes, other, prompt] = await withRun(memory, async (session) => {
    expect("openFrame" in session).toBe(false);
    const notesBatch = await session.learn(new LearnContext("run-1", "notes", [noted]));
    const otherBatch = await session.learn(new LearnContext("run-1", "other", [noted]));
    session.commit("one", notesBatch);
    return [notesBatch, otherBatch, promptOf(model.requests.at(-1)!)] as const;
  });
  expect(prompt.split("\n")[0]).toBe("Memory: other");
  expect(prompt).not.toContain("Frame:");
  const material = JSON.parse(prompt.split("Episodes: ")[1]!);
  expect(material.episodes[0].memory_sections).toEqual([{ memory: "notes", text: "prior" }]);
  expect(material.episodes[0].occurrence_path).toEqual([]);
  expect(notes[0]!.mutationId).not.toBe(other[0]!.mutationId);
  await withRun(memory, async (session) => expect(await session.recall()).toBe("[a] Cite"));
});

test("close while learn awaits the model is busy, and a session closed meanwhile fails the learn", async () => {
  let release!: (model: ChatModel) => void;
  const pending = new Promise<ChatModel>((resolve) => {
    release = resolve;
  });
  const memory = newMemory({ llmConfig: OLLAMA, createModel: () => pending });
  const session = memory.openRun();
  const learning = session.learn(context());
  const busy = failure(() => memory.close());
  expect([busy.code, busy.retryable]).toEqual(["reflective.busy", true]);
  session.close();
  release(new ScriptedModel([reply(NOOP)], OLLAMA));
  const error = await rejection(learning);
  expect([error.stage, error.code]).toEqual(["session", "reflective.session_closed"]);
  memory.close();
});

test("an episode holding Infinity reaches the prompt as Infinity", async () => {
  const model = installModel([reply(NOOP)]);
  const memory = newMemory({ llmConfig: OLLAMA, createModel: model.createModel });
  const infinite = new MemoryEpisode("run-1", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { text: "hi" }, {
    output: { score: Infinity },
  });
  await withRun(memory, (session) => session.learn(new LearnContext("run-1", "reflective", [infinite])));
  expect(promptOf(model.requests[0]!)).toContain('"output":{"score":Infinity}');
});

test("REFLECTION_SCHEMA is the frozen reply schema", () => {
  expect(REFLECTION_SCHEMA.title).toBe("_ReflectionResponse");
  expect((REFLECTION_SCHEMA.properties as Mutable).actions.minItems).toBe(1n);
  expect(Object.isFrozen(REFLECTION_SCHEMA)).toBe(true);
  expect(Object.isFrozen(REFLECTION_SCHEMA.properties)).toBe(true);
});

test("boundedJsonMaterial counts code points and keeps both ends", () => {
  expect(boundedJsonMaterial({ a: [1n, Infinity] }, 256)).toBe('{"a":[1,Infinity]}');
  const text = "😀".repeat(300);
  const material = boundedJsonMaterial(text, 256);
  expect(codePoints(material)).toBeLessThanOrEqual(256);
  const summary = JSON.parse(material);
  expect(Object.keys(summary)).toEqual(["original_chars", "prefix", "sha256", "suffix", "truncated"]);
  expect(summary.original_chars).toBe(302);
  expect(summary.prefix.startsWith('"😀')).toBe(true);
  expect(summary.suffix.endsWith('😀"')).toBe(true);
  expect(summary.prefix.isWellFormed() && summary.suffix.isWellFormed()).toBe(true);
  expect(summary.sha256).toBe(createHash("sha256").update(JSON.stringify(text)).digest("hex"));
  expect(summary.truncated).toBe(true);
});

/** The summary boundedJsonMaterial gives, cut from an array of the material's code points. */
function arrayBoundedMaterial(value: JsonObject | string, limit: number): string {
  const material = writeAsJson(value as Value);
  const chars = Array.from(material);
  if (chars.length <= limit) return material;
  const digest = createHash("sha256").update(material).digest("hex");
  const summary = (chunk: number) =>
    writeAsJson({
      original_chars: BigInt(chars.length),
      prefix: chars.slice(0, chunk).join(""),
      sha256: digest,
      suffix: chunk ? chars.slice(-chunk).join("") : "",
      truncated: true,
    });
  let chunk = Math.max(0, Math.floor((limit - 200) / 2));
  let encoded = summary(chunk);
  while (Array.from(encoded).length > limit && chunk) {
    const reduction = Math.max(1, Math.floor((Array.from(encoded).length - limit + 1) / 2));
    chunk = Math.max(0, chunk - reduction);
    encoded = summary(chunk);
  }
  return encoded;
}

test("boundedJsonMaterial and boundedJsonText cut the same text as a code-point array would", () => {
  // Odd-length runs put surrogate pairs on both sides of every cut point across the limits.
  const values: (JsonObject | string)[] = [
    "a😀é".repeat(700),
    `x${"😀".repeat(1500)}`,
    { lone: "\ud800 then \udc00", runs: ["é😀".repeat(333), `😀${"ab😀".repeat(401)}`] },
    "y".repeat(3000),
  ];
  for (const value of values) {
    for (const limit of [0, 1, 199, 200, 201, 250, 300, 301, 1000, 5000]) {
      const material = boundedJsonMaterial(value, limit);
      expect(material).toBe(arrayBoundedMaterial(value, limit));
      expect(boundedJsonText(writeAsJson(value as Value), limit)).toBe(material);
      expect(material.isWellFormed()).toBe(true);
      if (limit >= 300) expect(codePoints(material)).toBeLessThanOrEqual(limit);
    }
  }
});

// Needs Bun.gc to settle the heap before each sample, so it does not run under the Node suite.
test.skipIf(typeof Bun === "undefined")("boundedJsonMaterial keeps no per-code-point copy of long material", () => {
  // Above Latin-1, so each code point would be its own string rather than a cached one.
  const text = "é😀x".repeat(1_000_000);
  Bun.gc(true);
  const before = process.memoryUsage().heapUsed;
  let peak = before;
  const realHash = crypto.createHash;
  // Sampled once the length is known and before the summary is written.
  const hashSpy = spyOn(crypto, "createHash").mockImplementation(((...args: Parameters<typeof crypto.createHash>) => {
    Bun.gc(true);
    peak = Math.max(peak, process.memoryUsage().heapUsed);
    return Reflect.apply(realHash, crypto, args) as ReturnType<typeof crypto.createHash>;
  }) as typeof crypto.createHash);
  let material: string;
  try {
    material = boundedJsonMaterial(text, 1000);
  } finally {
    hashSpy.mockRestore();
  }
  // The material itself is about 8 MB; an array of its code points is several times that.
  expect(peak - before).toBeLessThan(32 * 1024 * 1024);
  expect(JSON.parse(material).original_chars).toBe(3_000_002);
});

test("visibleInteractionJson writes each interaction with sorted keys", () => {
  expect(Object.keys(visibleInteractionJson(new ToolCallInteraction("search", { q: "x" })))).toEqual([
    "arguments",
    "kind",
    "name",
  ]);
  expect(visibleInteractionJson(new ToolResultInteraction("search", "found"))).toEqual({
    kind: "tool_result",
    name: "search",
    text: "found",
  });
  expect(visibleInteractionJson(new ModelTextInteraction("hi"))).toEqual({ kind: "model_text", text: "hi" });
  expect(visibleInteractionJson(new ModelValueInteraction([1n]))).toEqual({ kind: "model_value", value: [1n] });
  expect(() => visibleInteractionJson({} as never)).toThrow(new TypeError("unsupported visible interaction"));
});
