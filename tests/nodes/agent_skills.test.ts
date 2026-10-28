/**
 * AGENT skill grants: the grant checks, the skills listing in the system prompt, `load_skill` on
 * demand, and only nodes with the skills trait receiving skills.
 */

import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, Response, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { type AgentPrompt, AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { Skill, SkillError } from "../../src/skills/catalog.ts";
import { LOAD_SKILL, systemPromptWithSkills } from "../../src/skills/tool.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, nodeFlow, serving, textReply } from "./run_state_fixtures.ts";

/** The node's private render, which `forward` and `learningMaterial` share. */
function renderPrompt(
  node: AgentNode,
  inputs: Readonly<Record<string, unknown>>,
  skills: ReadonlyMap<string, Skill> | null,
): AgentPrompt {
  type Inputs = Readonly<Record<string, unknown>>;
  type Render = { renderPrompt(inputs: Inputs, skills: ReadonlyMap<string, Skill> | null): AgentPrompt };
  return (node as unknown as Render).renderPrompt(inputs, skills);
}

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

// The `review` skill's directory, with one bundled file; the path is fixed here, the files exist
// only while this file's tests run.
const ROOT = join(tmpdir(), `yian-skills-${crypto.randomUUID()}`);
const REVIEW_DIR = join(ROOT, "review");
beforeAll(() => {
  mkdirSync(join(REVIEW_DIR, "references"), { recursive: true });
  writeFileSync(join(REVIEW_DIR, "references", "style.md"), "Prefer small diffs.\n");
});
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/** The failure message of an evaluation outcome; fails the test when it isn't a failure. */
function failureMessage(outcome: unknown): string {
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure.message;
}

const SKILLS: ReadonlyMap<string, Skill> = new Map([
  [
    "review",
    new Skill({
      name: "review",
      description: "Review code changes.",
      instructions: "Read references/style.md first.",
      directory: REVIEW_DIR,
    }),
  ],
  ["deploy", new Skill({ name: "deploy", description: "Ship a release.", instructions: "Tag, then publish." })],
  ["other", new Skill({ name: "other", description: "Unrelated.", instructions: "Not granted." })],
]);

test.each([
  ["a load_skill tool", { tools: [LOAD_SKILL] }, "reserved tool name"],
  ["a load_skill tool beside a grant", { skills: ["review"], tools: [LOAD_SKILL] }, "reserved tool name"],
  ["a duplicate", { skills: ["review", "review"] }, 'duplicate skill grant in ["review","review"]'],
  ["an invalid name", { skills: ["Review"] }, "invalid skill name 'Review'"],
  ["a string", { skills: "some" as "all" }, "list of names or 'all'"],
] as const)("agent rejects invalid skill grants: %s", (_label, options, message) => {
  expect(() => new AgentNode("agent", { prompt: "p", ...options })).toThrow(SkillError);
  expect(() => new AgentNode("agent", { prompt: "p", ...options })).toThrow(message);
});

test("agent skill grants default to none", () => {
  expect(new AgentNode("agent").skills).toEqual([]);
  expect(new AgentNode("agent", { skills: "all" }).skills).toBe("all");
});

test("a missing grant is refused", () => {
  const node = new AgentNode("agent", { prompt: "p", skills: ["absent"] });
  expect(() => node.grantedSkills(SKILLS)).toThrow(SkillError);
  expect(() => node.grantedSkills(SKILLS)).toThrow(
    new SkillError(`AGENT node 'agent': granted skills not supplied by the host: ["absent"]`),
  );
});

test("a grant without a skills mapping is refused", () => {
  const node = new AgentNode("agent", { prompt: "p", skills: "all" });
  expect(() => node.grantedSkills(null)).toThrow(SkillError);
  expect(() => node.grantedSkills(null)).toThrow(
    new SkillError("AGENT node 'agent' grants skills, but the host supplied no skills= mapping"),
  );
});

test("grantedSkills selects the grants sorted by name", () => {
  expect(new AgentNode("agent").grantedSkills(null)).toEqual([]);
  const names = (node: AgentNode) => node.grantedSkills(SKILLS).map((skill) => skill.name);
  expect(names(new AgentNode("agent", { skills: ["review", "deploy"] }))).toEqual(["deploy", "review"]);
  expect(names(new AgentNode("agent", { skills: "all" }))).toEqual(["deploy", "other", "review"]);
  expect(new AgentNode("agent", { skills: ["review"] }).grantedSkills(SKILLS)[0]).toBe(SKILLS.get("review")!);
});

test("only nodes with the skills trait receive skills", async () => {
  /** Record the capabilities one evaluation supplies. */
  class Probe extends Node {
    readonly kind = NodeKind.CODE;
    seen: NodeCaps = {};

    override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
      this.seen = caps;
      return new Output("ok");
    }
  }
  class SkillsProbe extends Probe {
    override readonly needsSkills = true;
  }
  const plain = new Probe("code");
  const optedIn = new SkillsProbe("skilled");

  expect((await evalNode(plain, null, new VariablePool(), { skills: SKILLS })).outcome).toEqual(new Output("ok"));
  expect((await evalNode(optedIn, null, new VariablePool(), { skills: SKILLS })).outcome).toEqual(new Output("ok"));

  expect("skills" in plain.seen).toBe(false);
  expect(optedIn.seen.skills).toBe(SKILLS);
});

test("a missing grant fails the node", async () => {
  const models = serving(new ScriptedModel([], CONFIG));
  const outcome = await drive(new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, skills: ["absent"] }), {
    models,
    skills: SKILLS,
  });
  expect(failureMessage(outcome)).toContain('not supplied by the host: ["absent"]');
  expect(models.built).toBe(0);
});

test("a grant without a skills mapping fails the node", async () => {
  const models = serving(new ScriptedModel([], CONFIG));
  const outcome = await drive(new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, skills: "all" }), { models });
  expect(failureMessage(outcome)).toContain("the host supplied no skills= mapping");
  expect(models.built).toBe(0);
});

test("no grant or no supply keeps the system prompt byte for byte", async () => {
  const model = new ScriptedModel([textReply("a"), textReply("b")], CONFIG);
  const models = serving(model);
  await drive(new AgentNode("no-grant", { prompt: "p", llmConfig: CONFIG }), { models, skills: SKILLS });
  await drive(new AgentNode("all-grant", { prompt: "p", llmConfig: CONFIG, skills: "all" }), { models, skills: new Map() });
  expect(model.requests.map((request) => request.messages[0]!.text())).toEqual([DEFAULT_SYSTEM, DEFAULT_SYSTEM]);
});

test("the memory context gets the system prompt without the listing", async () => {
  const seen: string[] = [];
  const openMemoryContext = async (system: string) => {
    seen.push(system);
    return new MemoryContext();
  };
  const node = new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, skills: ["review"] });
  const model = new ScriptedModel([textReply("ok")], CONFIG);
  const models = serving(model);
  expect(await node.forward({}, { openMemoryContext, skills: SKILLS, models })).toEqual(new Output("ok"));
  expect(seen).toEqual([DEFAULT_SYSTEM]);
  expect(model.requests[0]!.messages[0]!.text()).toBe(systemPromptWithSkills(DEFAULT_SYSTEM, [SKILLS.get("review")!]));
});

/** A reply calling `load_skill` with `args` as call `id`. */
function loadCall(args: JsonObject, id = "1"): Response {
  const call = new ToolCallPart({ id, name: LOAD_SKILL, arguments: args });
  return new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
}

/** The content of the tool result ending `messages`. */
function lastResult(messages: readonly Message[]): string {
  return (messages.at(-1)!.parts[0] as ToolResultPart).content;
}

test("load_skill is not callable without a grant", async () => {
  const outcome = await drive(new AgentNode("agent", { prompt: "p", llmConfig: CONFIG }), {
    models: serving(new ScriptedModel([loadCall({ name: "review" })], CONFIG)),
  });
  expect(failureMessage(outcome)).toContain("requested tool 'load_skill', but it is not granted");
});

test("all grants every supplied skill", async () => {
  const model = new ScriptedModel([textReply("ok")], CONFIG);
  const node = new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, skills: "all" });
  expect(await drive(node, { models: serving(model), skills: SKILLS })).toEqual(new Output("ok"));
  expect(model.requests[0]!.messages[0]!.text()).toBe(systemPromptWithSkills(DEFAULT_SYSTEM, [...SKILLS.values()]));
});

test("granted skills are listed and loaded on demand", async () => {
  const model = new ScriptedModel(
    [
      loadCall({ name: "review" }, "a"),
      loadCall({ name: "review", path: "references/style.md" }, "b"),
      textReply("reviewed"),
    ],
    CONFIG,
  );
  const node = new AgentNode("agent", { prompt: "Review it", llmConfig: CONFIG, skills: ["review", "deploy"] });

  const evaluated = await evalNode(node, nodeFlow(node), new VariablePool(), {
    models: serving(model),
    skills: SKILLS,
  });

  expect(evaluated.outcome).toEqual(new Output("reviewed"));
  const expectedSystem = systemPromptWithSkills(DEFAULT_SYSTEM, [SKILLS.get("deploy")!, SKILLS.get("review")!]);
  expect(model.requests[0]!.messages[0]!.text()).toBe(expectedSystem);
  expect(expectedSystem).toContain("- deploy: Ship a release.\n- review: Review code changes.");
  expect(expectedSystem).not.toContain("other");
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual([LOAD_SKILL]);
  expect(lastResult(model.requests[1]!.messages)).toBe("Read references/style.md first.");
  expect(lastResult(model.requests[2]!.messages)).toBe("Prefer small diffs.\n");
  // Learning renders the same system prompt from the skills the run recorded.
  expect(renderPrompt(node, {}, SKILLS).baseSystemPrompt).toBe(expectedSystem);
});

test("skills with structured output still run the tool loop", async () => {
  const model = new ScriptedModel([loadCall({ name: "review" }), textReply("seven"), textReply('{"value": 7}')], CONFIG);
  const node = new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, skills: ["review"] });
  node.outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));

  expect(await drive(node, { models: serving(model), skills: SKILLS })).toEqual(new Output(7n));
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual([LOAD_SKILL]);
  expect(model.requests).toHaveLength(3);
  const structured = model.requests[2]!.messages;
  expect(structured.some((message) => message.role === "tool" && lastResult([message]) === "Read references/style.md first.")).toBe(
    true,
  );
});

test("an escape attempt fails the node", async () => {
  const model = new ScriptedModel([loadCall({ name: "review", path: "../review/../../secret" })], CONFIG);
  const node = new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, skills: ["review"] });
  const outcome = await drive(node, { models: serving(model), skills: SKILLS });
  expect(failureMessage(outcome)).toContain("relative path inside the skill directory");
});
