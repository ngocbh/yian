/**
 * `runFlow`/`resumeFlow` `skills`: early checks and durable resume with on-demand skills; a port
 * of `tests/engine/test_run_skills.py`.
 *
 * Ported: every test, through `loadFlow`, `runFlow` and `resumeFlow` (`runWith`, `resumeWith`),
 * with a `ScriptedModel` serving every AGENT and the skills read by `loadSkills` from a temp root.
 *
 * Changed:
 * - `FLOW` has `llm_config: {provider: openai, model: test-model}`, since a run refuses an AGENT
 *   without a provider at start (Python patches the model builder instead).
 * - The early checks throw a `RangeError` where Python raises `ValueError`; a skills mapping is a
 *   `Map`, a hand-made skill a `new Skill({...})`; a list of names in a message is written in the
 *   text form (`["release-notes"]`, Python `['release-notes']`).
 * - A crash is a frozen log: `CrashAt` marks request N, and from then on every durable log's
 *   append throws a `Crash` the run rejects with (Python: the model raises `_SimulatedCrash` for
 *   request N). Either way no record of request N's outcome reaches the log.
 *
 * `run.test.ts`'s TS-only skill rows stay: they pin the full messages of the same checks.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { loadSkills } from "../../src/compose/skills.ts";
import type { EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, ToolCallPart, type ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { JsonlRunLog, NodeStartedRecord, NodeSucceededRecord, readRunLog, type RunRecord } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { LOAD_SKILL, systemPromptWithSkills } from "../../src/skills/tool.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Crash, rejection, textReply } from "./run_state_fixtures.ts";
import { effectsOf, resumeWith, runWith, SESSION_CONFIG } from "./session_fakes.ts";

/** A `ScriptedModel` that sets `frozen` as it serves request `at` (1-based); `crashes` reads it. */
class CrashAt extends ScriptedModel {
  readonly at: number;
  frozen = false;

  constructor(replies: ScriptedReply[], at: number) {
    super(replies, SESSION_CONFIG);
    this.at = at;
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    if (this.requests.length === this.at - 1) this.frozen = true;
    return super.stream(request, options);
  }
}

/** Await `run`, which must reject with the `Crash` every durable log throws once `model` froze them. */
async function crashes(model: CrashAt, run: () => Promise<unknown>): Promise<void> {
  const crash = new Crash("process lost");
  const append = JsonlRunLog.prototype.append;
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord) {
    if (model.frozen) throw crash;
    return append.call(this, record);
  };
  try {
    expect(await rejection(run())).toBe(crash);
  } finally {
    JsonlRunLog.prototype.append = append;
  }
}

/** A reply calling `load_skill` with `args` as call `id`. */
function loadCall(args: JsonObject, id: string): Response {
  const call = new ToolCallPart({ id, name: LOAD_SKILL, arguments: args });
  return new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
}

/** The system text of `model`'s first request. */
function systemText(model: ScriptedModel): string {
  return model.requests[0]!.messages[0]!.text();
}

/** The content of the tool result ending `model`'s request `index`. */
function lastResult(model: ScriptedModel, index: number): string {
  return (model.requests[index]!.messages.at(-1)!.parts[0] as ToolResultPart).content;
}

const FLOW = `id: skills-e2e
name: skills-e2e
llm_config: {provider: openai, model: test-model}
input:
  change: !str
nodes:
  review:
    kind: agent
    input:
      change: \${input.change}
    prompt: "Review \${change}"
    skills: [code-review, release-notes]
    output: str
output: \${review.output}
`;

const SESSION_FLOW = FLOW.replace(
  "    skills: [code-review, release-notes]\n",
  "    long_term_memory: session\n    skills: [code-review, release-notes]\n",
);

const ALL_FLOW = FLOW.replace("skills: [code-review, release-notes]", "skills: all");

const NESTED = `id: nested
name: nested
defs:
  body:
    nodes:
      inner:
        kind: agent
        prompt: hi
        skills: [code-review]
        output: str
    output: \${inner.output}
nodes:
  outer:
    kind: call
    call: body
output: \${outer.output}
`;

/** Write skill `name` under `root` and return its directory. */
function writeSkill(root: string, name: string, description: string, body: string): string {
  const directory = join(root, name);
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, "SKILL.md"), `---\nname: ${name}\ndescription: ${description}\nlicense: MIT\n---\n\n${body}`);
  return directory;
}

let dir: string;
let skillRoot: string;
let log: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-skills-"));
  skillRoot = join(dir, "skills");
  log = join(dir, "run.jsonl");
  const review = writeSkill(skillRoot, "code-review", "Review a diff for bugs.", "Read references/checklist.md.\n");
  mkdirSync(join(review, "references"));
  writeFileSync(join(review, "references", "checklist.md"), "1. Tests pass.\n");
  writeSkill(skillRoot, "release-notes", "Write release notes.", "Group by area.\n");
  writeSkill(skillRoot, "unused", "Not granted.", "Never shown.\n");
  writeFileSync(join(dir, "secret.txt"), "do not leak\n");
});

afterEach(() => rmSync(dir, { recursive: true, force: true }));

test("runFlow delivers skills to the agent", async () => {
  const model = new ScriptedModel([textReply("ok")], SESSION_CONFIG);
  const skills = loadSkills([skillRoot]);
  const result = await runWith(await loadFlow(FLOW), { change: "x" }, null, model, { skills });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  const granted = [skills.get("code-review")!, skills.get("release-notes")!];
  expect(systemText(model)).toBe(systemPromptWithSkills(DEFAULT_SYSTEM, granted));
});

test("load_skill runs through a session conversation", async () => {
  const model = new ScriptedModel([loadCall({ name: "code-review" }, "a"), textReply("done")], SESSION_CONFIG);
  const memory = new SessionMemory();
  const result = await runWith(await loadFlow(SESSION_FLOW), { change: "x" }, memory, model, {
    skills: loadSkills([skillRoot]),
  });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(lastResult(model, 1)).toBe("Read references/checklist.md.\n");
  expect(effectsOf(memory)).toEqual([
    ["model", "completed"],
    ["tool", "completed"],
    ["model", "completed"],
  ]);
});

test("a missing skill is rejected before any record", async () => {
  const model = new ScriptedModel([], SESSION_CONFIG);
  const skills = new Map([["code-review", new Skill({ name: "code-review", description: "d", instructions: "" })]]);
  const error = await rejection(runWith(await loadFlow(FLOW), { change: "x" }, null, model, { log, skills }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(`AGENT node 'review' grants skills not supplied to this run: ["release-notes"]`);
  expect(existsSync(log)).toBe(false);

  const nested = await rejection(runWith(await loadFlow(NESTED), {}, null, model, { skills: new Map() }));
  expect(nested).toBeInstanceOf(RangeError);
  expect((nested as Error).message).toContain("'outer/inner' grants skills not supplied");
});

test.each([
  ["listed", FLOW],
  ["all", ALL_FLOW],
  ["nested", NESTED],
])("a granting flow needs a skills mapping: %s", async (_label, text) => {
  const inputs = text === NESTED ? {} : { change: "x" };
  const error = await rejection(runWith(await loadFlow(text), inputs, null, new ScriptedModel([], SESSION_CONFIG), { log }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("grants skills, but no skills= mapping was passed");
  expect(existsSync(log)).toBe(false);
});

test("skills: all with an empty mapping grants none", async () => {
  const model = new ScriptedModel([textReply("ok")], SESSION_CONFIG);
  const result = await runWith(await loadFlow(ALL_FLOW), { change: "x" }, null, model, { skills: new Map() });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(systemText(model)).toBe(DEFAULT_SYSTEM);
});

test("a resume of a skills: all flow needs the mapping", async () => {
  const crashing = new CrashAt([textReply("never recorded")], 1);
  await crashes(crashing, async () =>
    runWith(await loadFlow(ALL_FLOW), { change: "x" }, null, crashing, { log, skills: loadSkills([skillRoot]) }),
  );
  const before = readFileSync(log);

  const refused = await rejection(resumeWith(await loadFlow(ALL_FLOW), log, null, new ScriptedModel([], SESSION_CONFIG)));
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("no skills= mapping was passed");
  expect(readFileSync(log)).toEqual(before);

  const model = new ScriptedModel([textReply("ok")], SESSION_CONFIG);
  const resumed = await resumeWith(await loadFlow(ALL_FLOW), log, null, model, { skills: loadSkills([skillRoot]) });
  expect(resumed.status).toBe("succeeded");
  expect(systemText(model)).toContain("- unused: Not granted.");
});

test("the skills mapping must be keyed by skill name", async () => {
  const skills = new Map([["alias", new Skill({ name: "code-review", description: "d", instructions: "" })]]);
  const error = await rejection(
    runWith(await loadFlow(FLOW), { change: "x" }, null, new ScriptedModel([], SESSION_CONFIG), { skills }),
  );
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("skills['alias'] must be a Skill named 'alias'");
});

test("end to end: listing, load, escape and resume", async () => {
  const skills = loadSkills([skillRoot]);
  const first = new CrashAt(
    [
      loadCall({ name: "code-review" }, "a"),
      loadCall({ name: "code-review", path: "references/checklist.md" }, "b"),
      textReply("never recorded"),
    ],
    3,
  );
  await crashes(first, async () =>
    runWith(await loadFlow(FLOW), { change: "diff-1" }, null, first, { log, runId: "skills-e2e", skills }),
  );

  const system = systemText(first);
  expect(system).toContain("- code-review: Review a diff for bugs.\n- release-notes: Write release notes.");
  expect(system).not.toContain("unused");
  expect(lastResult(first, 1)).toBe("Read references/checklist.md.\n");
  expect(lastResult(first, 2)).toBe("1. Tests pass.\n");

  const before = readFileSync(log);
  const refused = await rejection(resumeWith(await loadFlow(FLOW), log, null, new ScriptedModel([], SESSION_CONFIG)));
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("no skills= mapping was passed");
  expect(readFileSync(log)).toEqual(before);

  const second = new ScriptedModel(
    [
      loadCall({ name: "code-review" }, "a"),
      loadCall({ name: "code-review", path: "references/checklist.md" }, "b"),
      textReply("looks good"),
    ],
    SESSION_CONFIG,
  );
  const resumed = await resumeWith(await loadFlow(FLOW), log, null, second, { skills: loadSkills([skillRoot]) });

  expect(resumed.failure).toBeNull();
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("looks good");
  expect(second.requests[2]!.messages).toEqual(first.requests[2]!.messages);
  const records = readRunLog(log).map((envelope) => envelope.record);
  const review = (kind: typeof NodeStartedRecord | typeof NodeSucceededRecord) =>
    records.filter((record) => record instanceof kind && record.nodeId === "review").length;
  expect(review(NodeStartedRecord)).toBe(2);
  expect(review(NodeSucceededRecord)).toBe(1);

  const escape = new ScriptedModel([loadCall({ name: "code-review", path: "../../secret.txt" }, "c")], SESSION_CONFIG);
  const failed = await runWith(await loadFlow(FLOW), { change: "diff-2" }, null, escape, { skills });
  expect(failed.status).toBe("failed");
  expect(failed.failure!.message).toContain("relative path inside the skill directory");
  expect(failed.failure!.message).not.toContain("do not leak");
});
