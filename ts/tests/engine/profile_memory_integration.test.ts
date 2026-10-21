/**
 * Public runs, backward attempts and recovery with `ProfileMemory`: mirrors
 * `tests/engine/test_profile_memory_integration.py`.
 *
 * Ported, through `runFlow`: `test_sibling_agents_recall_one_shared_profile_document`; through
 * `runFlow` and `RunResult.backward`: `test_recall_learn_and_commit_through_a_public_run`,
 * `test_nested_declarations_of_one_key_show_and_rewrite_the_document_once`,
 * `test_profile_notes_and_a_session_transcript_share_one_agent_context`,
 * `test_human_edit_during_learning_fails_the_commit_without_a_model` and
 * `test_a_new_attempt_after_a_mismatch_learns_from_the_edited_document`; through `resumeFlow` too:
 * `test_resume_reopens_the_recorded_version_for_backward` and
 * `test_an_attempt_killed_after_its_write_is_unknown_and_findable_in_provenance`.
 *
 * Changed: one `ScriptedModel` serves a run's AGENTs, its forward replies and then its AGENT
 * backward replies, where Python builds a chat per AGENT and patches a fresh one in for backward;
 * the run's environment names the default model (conftest's in Python), since the flow names none.
 * The profile learner is the backend's `createModel` option, reading the test's current `Learner`
 * (Python patches `profile.model_from_config`). The killed attempt's store throws a plain `Error`
 * after its write, a host failure that rejects `backward` (Python raises `SystemExit`).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { EventStream } from "../../src/llm/events.ts";
import type { Request } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { JsonObject } from "../../src/memory/contracts.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";
import { JsonlRunLog, MemoryCommitFailedRecord, RunStartedRecord } from "../../src/runtime/log.ts";
import { rejected, reply, resumeWith, runWith, sentTexts } from "./session_fakes.ts";

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
    long_term_memory: session
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
output: \${reply.output}
`;

const LEARNER_CONFIG = { provider: "ollama", model: "local-profiler" };

/** Stub profile learner that can await a side effect before replying (Python's `_Learner`). */
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
    async function* events() {
      await before!();
      yield* inner;
    }
    return new EventStream(events(), () => inner.close());
  }
}

let dir: string;
let env: Env;
// The learner every profile backend builds, and the configs it was built from (Python's `_learner`).
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
function backend(store: FileMarkdownStore): ProfileMemory {
  return new ProfileMemory(store, {
    llmConfig: LEARNER_CONFIG,
    env,
    createModel: (config) => {
      built.push(config);
      return learner;
    },
  });
}

/** Python's `_profile`: Ada's document in a file store, learned by a local model. */
async function profile(): Promise<ProfileMemory> {
  const store = new FileMarkdownStore(join(dir, "ada.md"));
  await store.write("# Ada\n- Works on reefs.\n", { baseVersion: (await store.read())[1] });
  return backend(store);
}

/** An AGENT backward reply with no feedback (Python's `_reply()`). */
function noFeedback() {
  return reply(JSON.stringify({ input_feedback: {}, self_feedback: [], variable_feedback: {} }));
}

/** Python's `_forward`: a model answering `answers`, then `backwards` AGENT backward calls. */
function chat(answers: string[], backwards = 1): ScriptedModel {
  return new ScriptedModel([...answers.map(reply), ...Array.from({ length: backwards }, noFeedback)]);
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

test("profile notes and a session transcript share one agent context", async () => {
  const profileMemory = await profile();
  const memory = new Map<string, ProfileMemory | SessionMemory>([
    ["profile", profileMemory],
    ["session", new SessionMemory()],
  ]);
  const flow = await loadFlow(OFFICE);
  expect((await runWith(flow, { message: "hello" }, memory, chat(["First reply."], 0), { env })).status).toBe(
    "succeeded",
  );
  const model = chat(["Second reply."]);
  const result = await runWith(flow, { message: "again" }, memory, model, { learnable: true, env });
  expect(model.requests[0]!.messages[0]!.text().split("Works on reefs.")).toHaveLength(2);
  expect(sentTexts(model, 0)).toEqual(["hello", "First reply.", "again"]);
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
  using log = JsonlRunLog.resume(path);
  const failed = log
    .records()
    .map((envelope) => envelope.record)
    .filter((record) => record.constructor === MemoryCommitFailedRecord) as MemoryCommitFailedRecord[];
  expect(failed.map((record) => record.failure.code)).toEqual(["profile.version_mismatch"]);
});

test("resume reopens the recorded version for backward", async () => {
  const memory = await profile();
  const flow = await loadFlow(CHAT);
  const path = join(dir, "run.jsonl");
  const [recordedText, recordedVersion] = await memory.read();
  const registry = new Map([["profile", memory]]);
  await runWith(flow, { message: "hi" }, registry, chat(["Hello."], 0), { log: path, learnable: true, env });
  {
    using log = JsonlRunLog.resume(path);
    const started = log.records()[0]!.record as RunStartedRecord;
    expect(started.memorySnapshots).toHaveLength(1);
    expect(started.memorySnapshots![0]!.snapshot["profile_version"]).toBe(recordedVersion);
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
