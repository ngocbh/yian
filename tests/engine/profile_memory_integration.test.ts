/**
 * Public runs, backward attempts and recovery with `ProfileMemory`, and a host `MarkdownStore`
 * whose versions are git blob ids: the round trip, the `run_started` snapshot, the read of the
 * recorded version by a resumed run's backward, and a store that dropped it.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { type MarkdownStore, MarkdownVersionMismatch, ProfileMemory } from "../../src/index.ts";
import { EventStream } from "../../src/llm/events.ts";
import type { Request } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type JsonObject, MemoryOperationError, MemorySetupError } from "../../src/memory/contracts.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";
import {
  BackwardStartedRecord,
  JsonlRunLog,
  MemoryCommitFailedRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import { backwardReply, rejected, reply, resumeWith, runWith, sentTexts } from "./run_state_fixtures.ts";

const CHAT = `id: chat
name: chat
long_term_memory: profile
input:
  message: !str
nodes:
  reply:
    kind: agent
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
output: \${reply.output}
`;

const NESTED = `id: nested
name: nested
long_term_memory: profile
input:
  message: !str
defs:
  child:
    long_term_memory: profile
    input:
      message: !str
    nodes:
      reply:
        kind: agent
        long_term_memory: profile
        input:
          message: \${input.message}
        prompt: \${message}
        output: str
    output: \${reply.output}
nodes:
  invoke:
    kind: call
    call: child
    input:
      message: \${input.message}
output: \${invoke.output}
`;

const SIBLINGS = `id: siblings
name: siblings
input:
  message: !str
nodes:
  first:
    kind: agent
    long_term_memory: profile
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
  second:
    kind: agent
    long_term_memory: profile
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
output: \${first.output}
`;

const OFFICE = `id: office
name: office
long_term_memory: profile
input:
  message: !str
nodes:
  reply:
    kind: agent
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
output: \${reply.output}
`;

// Typed as `runFlow`'s `llmConfig`: a host passes ProfileMemory the same config it passes a run.
const LEARNER_CONFIG: Readonly<Record<string, unknown>> = {
  provider: "ollama",
  model: "local-profiler",
};

/** Stub profile learner that can await a side effect before replying. */
class Learner extends ScriptedModel {
  readonly before: (() => Promise<unknown>) | null;

  constructor(text: string, before: (() => Promise<unknown>) | null = null) {
    super([reply(`<profile>\n${text}\n</profile>`)]);
    this.before = before;
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    const inner = super.stream(request, options);
    const before = this.before;
    if (before === null) return inner;
    // A generator expression, not a hoisted declaration, so `before` stays narrowed to non-null.
    const events = async function* () {
      await before();
      yield* inner;
    };
    return new EventStream(events(), () => inner.close());
  }
}

let dir: string;
let env: Env;
// The learner every profile backend builds, and the configs it was built from.
let learner: Learner;
let built: Readonly<Record<string, unknown>>[];

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-profile-integration-"));
  const vars = new Map([
    ["YIAN_AUTH_FILE", join(dir, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "test-model"],
  ]);
  env = (name) => vars.get(name);
  learner = new Learner("unused");
  built = [];
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** A profile backend over `store`, learned by the test's current `learner`. */
function backend(store: MarkdownStore): ProfileMemory {
  return new ProfileMemory(store, {
    llmConfig: LEARNER_CONFIG,
    env,
    createModel: (config) => {
      built.push(config);
      return learner;
    },
  });
}

/** Ada's document in a file store, learned by a local model. */
async function profile(): Promise<ProfileMemory> {
  const store = new FileMarkdownStore(join(dir, "ada.md"));
  await store.write("# Ada\n- Works on reefs.\n", { baseVersion: (await store.read())[1] });
  return backend(store);
}

/** A model answering `answers`, then `backwards` AGENT backward calls. */
function chat(answers: string[], backwards = 1): ScriptedModel {
  return new ScriptedModel([...answers.map(reply), ...Array.from({ length: backwards }, () => backwardReply())]);
}

/** The JSON input of `model`'s learner request `index` (the last by default). */
function material(model: ScriptedModel, index = -1): JsonObject {
  return JSON.parse(model.requests.at(index)!.messages[1]!.text());
}

/** The provenance lines of the profile's write log. */
function provenance(): JsonObject[] {
  const lines = readFileSync(join(dir, ".ada.md.writes.jsonl"), "utf8").split("\n").filter((line) => line !== "");
  return lines.map((line) => JSON.parse(line).provenance);
}

test("recall, learn and commit through a public run", async () => {
  const memory = await profile();
  const model = chat(["Hello."]);
  const registry = new Map([["profile", memory]]);
  const result = await runWith(await loadFlow(CHAT), { message: "hi" }, registry, model, { learnable: true, env });
  expect(result.status).toBe("succeeded");
  expect(model.requests[0]!.messages[0]!.text()).toContain(JSON.stringify("# Ada\n- Works on reefs.\n"));

  learner = new Learner("# Ada\n- Works on reefs.\n- Prefers brief replies.");
  const outcome = await result.backward("Keep replies brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  expect(built).toHaveLength(1);
  expect(learner.requests).toHaveLength(1);
  const feedback = material(learner)["feedback"] as JsonObject[];
  expect(feedback.map((entry) => entry["value"])).toContain("Keep replies brief.");
  expect((await memory.read())[0]).toBe("# Ada\n- Works on reefs.\n- Prefers brief replies.\n");
  expect(provenance().at(-1)!["feedback"]).toEqual(["Keep replies brief."]);
});

test("nested declarations of one key show and rewrite the document once", async () => {
  const model = chat(["Hello."]);
  const registry = new Map([["profile", await profile()]]);
  const result = await runWith(await loadFlow(NESTED), { message: "hi" }, registry, model, { learnable: true, env });
  expect(result.status).toBe("succeeded");
  expect(model.requests[0]!.messages[0]!.text().split("Works on reefs.")).toHaveLength(2);
  learner = new Learner("# Ada\n- Brief.");
  expect((await result.backward("Be brief.")).status).toBe("succeeded");
  expect(learner.requests).toHaveLength(1);
});

test("sibling agents recall one shared profile document", async () => {
  const model = new ScriptedModel([reply("a"), reply("b")]);
  const memory = new Map([["profile", await profile()]]);
  const result = await runWith(await loadFlow(SIBLINGS), { message: "hi" }, memory, model, { env });
  expect(result.status).toBe("succeeded");
  const systems = model.requests.map((request) => request.messages[0]!);
  expect(systems).toHaveLength(2);
  for (const system of systems) {
    expect(system.role).toBe("system");
    expect(system.text().split("Works on reefs.")).toHaveLength(2);
  }
});

test("profile notes reach the agent context of every run", async () => {
  const profileMemory = await profile();
  const memory = new Map([["profile", profileMemory]]);
  const flow = await loadFlow(OFFICE);
  expect((await runWith(flow, { message: "hello" }, memory, chat(["First reply."], 0), { env })).status).toBe(
    "succeeded",
  );
  const model = chat(["Second reply."]);
  const result = await runWith(flow, { message: "again" }, memory, model, { learnable: true, env });
  expect(model.requests[0]!.messages[0]!.text().split("Works on reefs.")).toHaveLength(2);
  expect(sentTexts(model, 0)).toEqual(["again"]);
  learner = new Learner("# Ada\n- Works on reefs.\n- Brief.");
  const outcome = await result.backward("Be brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  expect((await profileMemory.read())[0]).toBe("# Ada\n- Works on reefs.\n- Brief.\n");
});

test("human edit during learning fails the commit without a model", async () => {
  const memory = await profile();
  const path = join(dir, "run.jsonl");
  const registry = new Map([["profile", memory]]);
  const result = await runWith(await loadFlow(CHAT), { message: "hi" }, registry, chat(["Hello."]), {
    log: path,
    learnable: true,
    env,
  });

  const humanEdit = async () => memory.write("# Ada\n- Edited by Ada.\n", { baseVersion: (await memory.read())[1] });
  learner = new Learner("# Ada\n- From the model.", humanEdit);
  const outcome = await result.backward("Keep replies brief.");
  expect(outcome.status).toBe("failed");
  expect([outcome.failure!.stage, outcome.failure!.code]).toEqual(["commit", "profile.version_mismatch"]);
  expect((outcome.failure!.cause as { retryable: boolean }).retryable).toBe(true);
  expect(built).toHaveLength(1);
  expect(learner.requests).toHaveLength(1);
  expect((await memory.read())[0]).toBe("# Ada\n- Edited by Ada.\n");
  const log = JsonlRunLog.resume(path);
  try {
    const failed = log
      .records()
      .map((envelope) => envelope.record)
      .filter((record) => record.constructor === MemoryCommitFailedRecord) as MemoryCommitFailedRecord[];
    expect(failed.map((record) => record.failure.code)).toEqual(["profile.version_mismatch"]);
  } finally {
    log[Symbol.dispose]();
  }
});

test("resume reopens the recorded version for backward", async () => {
  const memory = await profile();
  const flow = await loadFlow(CHAT);
  const path = join(dir, "run.jsonl");
  const [recordedText, recordedVersion] = await memory.read();
  const registry = new Map([["profile", memory]]);
  await runWith(flow, { message: "hi" }, registry, chat(["Hello."], 0), { log: path, learnable: true, env });
  {
    const log = JsonlRunLog.resume(path);
    try {
      const started = log.records()[0]!.record as RunStartedRecord;
      expect(started.memorySnapshots).toHaveLength(1);
      expect(started.memorySnapshots![0]!.snapshot["profile_version"]).toBe(recordedVersion);
    } finally {
      log[Symbol.dispose]();
    }
  }
  await memory.write("# Ada\n- Edited later.\n", { baseVersion: recordedVersion });

  const reopened = new Map([["profile", backend(new FileMarkdownStore(join(dir, "ada.md")))]]);
  const resumed = await resumeWith(flow, path, reopened, chat([]), { env });
  learner = new Learner("# Ada\n- Edited later.\n- Brief.");
  const outcome = await resumed.backward("Be brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  const input = material(learner);
  expect(input["seen_profile"]).toBe(recordedText);
  expect(input["current_profile"]).toBe("# Ada\n- Edited later.\n");
  expect((await memory.read())[0]).toBe("# Ada\n- Edited later.\n- Brief.\n");
});

test("a new attempt after a mismatch learns from the edited document", async () => {
  const memory = await profile();
  const registry = new Map([["profile", memory]]);
  const result = await runWith(await loadFlow(CHAT), { message: "hi" }, registry, chat(["Hello."], 2), {
    learnable: true,
    env,
  });
  learner = new Learner("# Ada\n- From the model.", async () =>
    memory.write("# Ada\n- Edited by Ada.\n", { baseVersion: (await memory.read())[1] }),
  );
  expect((await result.backward("Keep replies brief.")).status).toBe("failed");

  learner = new Learner("# Ada\n- Edited by Ada.\n- Prefers brief replies.");
  const outcome = await result.backward("Keep replies brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  expect(material(learner)["current_profile"]).toBe("# Ada\n- Edited by Ada.\n");
  expect((await memory.read())[0]).toBe("# Ada\n- Edited by Ada.\n- Prefers brief replies.\n");
});

/** Store that writes, then fails like a process killed before the runtime records success. */
class KilledAfterWrite extends FileMarkdownStore {
  kill = true;

  override async write(text: string, options: Parameters<FileMarkdownStore["write"]>[1]): Promise<string> {
    const version = await super.write(text, options);
    if (this.kill && (options.provenance ?? null) !== null) throw new Error("killed after the store write");
    return version;
  }
}

test("an attempt killed after its write is unknown and findable in provenance", async () => {
  const store = new KilledAfterWrite(join(dir, "ada.md"));
  await store.write("# Ada\n", { baseVersion: (await store.read())[1] });
  const registry = new Map([["profile", backend(store)]]);
  const flow = await loadFlow(CHAT);
  const path = join(dir, "run.jsonl");
  await runWith(flow, { message: "hi" }, registry, chat(["Hello."], 0), { log: path, learnable: true, env });
  learner = new Learner("# Ada\n- Brief.");
  const killed = await resumeWith(flow, path, registry, chat([]), { env });
  expect(((await rejected(killed.backward("Be brief."))) as Error).message).toBe("killed after the store write");
  expect((await store.read())[0]).toBe("# Ada\n- Brief.\n");

  store.kill = false;
  const resumed = await resumeWith(flow, path, registry, chat([]), { env });
  const policies = resumed.backwardOutcome!.learning!.policies;
  expect(policies).toHaveLength(1);
  const unknown = policies[0]!;
  expect(unknown.status).toBe("unknown");
  expect(unknown.commitId!.startsWith("ac-memory-v3:")).toBe(true);
  const written = provenance().slice(1);
  expect(written.map((entry) => entry["commit_id"])).toEqual([unknown.commitId]);

  learner = new Learner("# Ada\n- Brief.");
  const outcome = await resumed.backward("Be brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  expect(learner.requests.at(-1)!.messages[0]!.text()).toContain("may already reflect this feedback");
  const input = material(learner);
  expect([input["seen_profile"], input["current_profile"]]).toEqual(["# Ada\n", "# Ada\n- Brief.\n"]);
  expect((await store.read())[0]).toBe("# Ada\n- Brief.\n");
  expect(provenance()).toHaveLength(2);
});

/** Git's blob id of `text`'s UTF-8 bytes: hex SHA-1 of `"blob <length>\0"` and the bytes. */
function blobId(text: string): string {
  return createHash("sha1").update(`blob ${Buffer.byteLength(text, "utf8")}\0${text}`, "utf8").digest("hex");
}

/** Hex SHA-256 of `text`'s UTF-8 bytes. */
function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** A git-like host store: versions are blob ids, and every version it handed out stays readable. */
class BlobStore implements MarkdownStore {
  readonly versions = new Map<string, string>();
  current: string;
  readonly reads: (string | null)[] = [];

  constructor() {
    // Seeded, so the empty document's version was really handed out.
    this.current = blobId("");
    this.versions.set(this.current, "");
  }

  // Synchronous, as a host may answer at once.
  read(version: string | null = null): readonly [string, string] {
    this.reads.push(version);
    if (version === null) return [this.versions.get(this.current)!, this.current];
    const text = this.versions.get(version);
    if (text === undefined) throw new RangeError("unknown version");
    return [text, version];
  }

  // Async, to cover the promise form.
  async write(text: string, { baseVersion }: { baseVersion: string }): Promise<string> {
    if (baseVersion !== this.current) throw new MarkdownVersionMismatch(baseVersion, this.current);
    this.current = blobId(text);
    this.versions.set(this.current, text);
    return this.current;
  }

  /** Drop `version`, as a store breaking the keep-every-version contract would. */
  forget(version: string): void {
    this.versions.delete(version);
  }
}

const SEED = "# Ada\n- Works on reefs.\n";
const EDITED = "# Ada\n- Edited later.\n";

test("a host store with blob id versions round-trips through a public run", async () => {
  expect(blobId("")).toBe("e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
  const store = new BlobStore();
  await store.write(SEED, { baseVersion: store.current });
  const memory = backend(store);
  const model = chat(["Hello."]);
  const registry = new Map([["profile", memory]]);
  const result = await runWith(await loadFlow(CHAT), { message: "hi" }, registry, model, { learnable: true, env });
  expect(model.requests[0]!.messages[0]!.text()).toContain(JSON.stringify(SEED));

  learner = new Learner("# Ada\n- Works on reefs.\n- Prefers brief replies.");
  const outcome = await result.backward("Keep replies brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  const updated = "# Ada\n- Works on reefs.\n- Prefers brief replies.\n";
  expect(store.versions.get(store.current)).toBe(updated);
  expect(store.current).toBe(blobId(updated));
  expect(store.current).toMatch(/^[0-9a-f]{40}$/);
  expect(await memory.read()).toEqual([updated, blobId(updated)]);
  for (const text of ["", SEED, updated]) {
    expect(await memory.read(blobId(text))).toEqual([text, blobId(text)]);
  }
});

test("the run started snapshot is a reference and a hash, never the text", async () => {
  const store = new BlobStore();
  await store.write(SEED, { baseVersion: store.current });
  const path = join(dir, "run.jsonl");
  const registry = new Map([["profile", backend(store)]]);
  await runWith(await loadFlow(CHAT), { message: "hi" }, registry, chat(["Hello."], 0), { log: path, env });
  const log = JsonlRunLog.resume(path);
  try {
    const started = log.records()[0]!.record as RunStartedRecord;
    expect(started.memorySnapshots).toHaveLength(1);
    expect(started.memorySnapshots![0]!.snapshot).toEqual({
      format: "yian.profile_memory",
      version: 1n,
      profile_version: blobId(SEED),
      text_sha256: sha256(SEED),
    });
  } finally {
    log[Symbol.dispose]();
  }
  // Only the first line: later lines, the AGENT's evidence, do hold the text.
  const first = readFileSync(path, "utf8").split("\n")[0]!;
  expect(JSON.parse(first).kind).toBe("run_started");
  expect(first).not.toContain("Works on reefs");
});

test("backward after resuming a finished run asks the host store for the recorded blob version", async () => {
  const store = new BlobStore();
  await store.write(SEED, { baseVersion: store.current });
  const flow = await loadFlow(CHAT);
  const path = join(dir, "run.jsonl");
  const registry = new Map([["profile", backend(store)]]);
  await runWith(flow, { message: "hi" }, registry, chat(["Hello."], 0), { log: path, learnable: true, env });
  await store.write(EDITED, { baseVersion: blobId(SEED) });
  store.reads.length = 0;

  const resumed = await resumeWith(flow, path, new Map([["profile", backend(store)]]), chat([]), { env });
  // A finished run opens no memory on resume; backward does.
  expect(store.reads).toEqual([]);
  learner = new Learner("# Ada\n- Edited later.\n- Brief.");
  const outcome = await resumed.backward("Be brief.");
  expect(outcome.failure).toBeNull();
  expect(outcome.status).toBe("succeeded");
  expect(store.reads).toEqual([blobId(SEED), null]);
  const input = material(learner);
  expect(input["seen_profile"]).toBe(SEED);
  expect(input["current_profile"]).toBe(EDITED);
  expect(store.versions.get(store.current)).toBe("# Ada\n- Edited later.\n- Brief.\n");
});

test("a host store that dropped the recorded version cannot reopen the run", async () => {
  const store = new BlobStore();
  await store.write(SEED, { baseVersion: store.current });
  const flow = await loadFlow(CHAT);
  const path = join(dir, "run.jsonl");
  const registry = new Map([["profile", backend(store)]]);
  await runWith(flow, { message: "hi" }, registry, chat(["Hello."], 0), { log: path, learnable: true, env });
  await store.write(EDITED, { baseVersion: blobId(SEED) });
  store.forget(blobId(SEED));

  const resumed = await resumeWith(flow, path, new Map([["profile", backend(store)]]), chat([]), { env });
  const error = await rejected(resumed.backward("Be brief."));
  expect(error).toBeInstanceOf(MemorySetupError);
  expect((error as MemorySetupError).message).toBe(
    "memory 'profile' could not open its run session: recorded profile version cannot be restored",
  );
  const cause = (error as MemorySetupError).cause as MemoryOperationError;
  expect(cause).toBeInstanceOf(MemoryOperationError);
  expect([cause.stage, cause.code]).toEqual(["openRun", "profile.snapshot_unavailable"]);
  expect(cause.cause).toBeInstanceOf(RangeError);
  expect((cause.cause as RangeError).message).toBe("unknown version");
  expect(resumed.backwardOutcome).toBeNull();
  const log = JsonlRunLog.resume(path);
  try {
    expect(log.records().some((envelope) => envelope.record.constructor === BackwardStartedRecord)).toBe(false);
  } finally {
    log[Symbol.dispose]();
  }
});
