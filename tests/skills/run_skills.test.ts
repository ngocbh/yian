/**
 * `runFlow`/`advance` `skills`: early checks and durable resume with on-demand skills. A crash is
 * a frozen log: `CrashAt` marks request N, and from then on every durable log's append throws a
 * `Crash` the run rejects with, so no record of request N's outcome reaches the log.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { loadSkills } from "../../src/compose/skills.ts";
import { Message, Response, ToolCallPart, type ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { NodeStartedRecord, NodeSucceededRecord, readRunLog } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { LOAD_SKILL, systemPromptWithSkills } from "../../src/skills/tool.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { CrashAt, crashes, PLAIN_CONFIG, rejected, resumeWith, runWith, textReply } from "./run_state_fixtures.ts";

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
  const model = new ScriptedModel([textReply("ok")], PLAIN_CONFIG);
  const skills = loadSkills([skillRoot]);
  const result = await runWith(await loadFlow(FLOW), { change: "x" }, null, model, { skills });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  const granted = [skills.get("code-review")!, skills.get("release-notes")!];
  expect(systemText(model)).toBe(systemPromptWithSkills(DEFAULT_SYSTEM, granted));
});

test("load_skill runs and the next request reads its result", async () => {
  const model = new ScriptedModel([loadCall({ name: "code-review" }, "a"), textReply("done")], PLAIN_CONFIG);
  const result = await runWith(await loadFlow(FLOW), { change: "x" }, null, model, { skills: loadSkills([skillRoot]) });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(model.requests).toHaveLength(2);
  expect(lastResult(model, 1)).toBe("Read references/checklist.md.\n");
});

test("a missing skill is rejected before any record", async () => {
  const model = new ScriptedModel([], PLAIN_CONFIG);
  const skills = new Map([["code-review", new Skill({ name: "code-review", description: "d", instructions: "" })]]);
  const error = await rejected(runWith(await loadFlow(FLOW), { change: "x" }, null, model, { log, skills }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(`AGENT node 'review' grants skills not supplied to this run: ["release-notes"]`);
  expect(existsSync(log)).toBe(false);

  const nested = await rejected(runWith(await loadFlow(NESTED), {}, null, model, { skills: new Map() }));
  expect(nested).toBeInstanceOf(RangeError);
  expect((nested as Error).message).toContain("'outer/inner' grants skills not supplied");
});

test.each([
  ["listed", FLOW],
  ["all", ALL_FLOW],
  ["nested", NESTED],
])("a granting flow needs a skills mapping: %s", async (_label, text) => {
  const inputs = text === NESTED ? {} : { change: "x" };
  const error = await rejected(runWith(await loadFlow(text), inputs, null, new ScriptedModel([], PLAIN_CONFIG), { log }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("grants skills, but no skills= mapping was passed");
  expect(existsSync(log)).toBe(false);
});

test("skills: all with an empty mapping grants none", async () => {
  const model = new ScriptedModel([textReply("ok")], PLAIN_CONFIG);
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

  const refused = await rejected(resumeWith(await loadFlow(ALL_FLOW), log, null, new ScriptedModel([], PLAIN_CONFIG)));
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("no skills= mapping was passed");
  expect(readFileSync(log)).toEqual(before);

  const model = new ScriptedModel([textReply("ok")], PLAIN_CONFIG);
  const resumed = await resumeWith(await loadFlow(ALL_FLOW), log, null, model, { skills: loadSkills([skillRoot]) });
  expect(resumed.status).toBe("succeeded");
  expect(systemText(model)).toContain("- unused: Not granted.");
});

test("the skills mapping must be keyed by skill name", async () => {
  const skills = new Map([["alias", new Skill({ name: "code-review", description: "d", instructions: "" })]]);
  const error = await rejected(
    runWith(await loadFlow(FLOW), { change: "x" }, null, new ScriptedModel([], PLAIN_CONFIG), { skills }),
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
  const refused = await rejected(resumeWith(await loadFlow(FLOW), log, null, new ScriptedModel([], PLAIN_CONFIG)));
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("no skills= mapping was passed");
  expect(readFileSync(log)).toEqual(before);

  // The two recorded calls and their `load_skill` results replay; only the third call is made.
  const second = new ScriptedModel([textReply("looks good")], PLAIN_CONFIG);
  const resumed = await resumeWith(await loadFlow(FLOW), log, null, second, { skills: loadSkills([skillRoot]) });

  expect(resumed.failure).toBeNull();
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("looks good");
  expect(second.requests[0]!.messages).toEqual(first.requests[2]!.messages);
  const records = readRunLog(log).map((envelope) => envelope.record);
  const review = (kind: typeof NodeStartedRecord | typeof NodeSucceededRecord) =>
    records.filter((record) => record instanceof kind && record.nodeId === "review").length;
  expect(review(NodeStartedRecord)).toBe(2);
  expect(review(NodeSucceededRecord)).toBe(1);

  const escape = new ScriptedModel([loadCall({ name: "code-review", path: "../../secret.txt" }, "c")], PLAIN_CONFIG);
  const failed = await runWith(await loadFlow(FLOW), { change: "diff-2" }, null, escape, { skills });
  expect(failed.status).toBe("failed");
  expect(failed.failure!.message).toContain("relative path inside the skill directory");
  expect(failed.failure!.message).not.toContain("do not leak");
});
