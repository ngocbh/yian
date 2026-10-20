/**
 * AGENT skill grants: mirrors `tests/engine/test_agent_skills.py`.
 *
 * Ported: `test_agent_rejects_invalid_skill_grants` (5 rows) and
 * `test_agent_skill_grants_default_to_none`; `test_missing_grant_fails_the_node` and
 * `test_grant_without_a_skills_mapping_fails_the_node`, each as its `grantedSkills` half and its
 * node-failure half through `evalNode` (`drive`), which also checks that no model is built;
 * `test_no_grant_or_no_supply_keeps_the_system_prompt_byte_for_byte` and
 * `test_load_skill_is_not_callable_without_a_grant` through `drive`, with a `ScriptedModel` from
 * `models`.
 *
 * Ported (adapted): `test_only_nodes_with_the_skills_trait_receive_skills`, on a local probe that
 * records its capabilities where Python's `FuncNode` would refuse an unexpected keyword: a node
 * without `needsSkills` gets no `skills`, and one with it gets the host's map itself.
 *
 * Changed: the grant checks are called directly where Python runs the node through `eval_node`;
 * a list of names in a message is written in the text form (`["absent"]`, Python
 * `['absent']`). `test_memory_context_gets_the_system_prompt_without_the_listing`: only its memory
 * half, the opener seeing `DEFAULT_SYSTEM` without the listing; the node then stops, since a skill
 * grant needs the tool loop, so the model-request half is Later (T5-11b).
 *
 * TS-only: `grantedSkills` returns no skills without a grant, and the granted ones sorted by
 * name.
 *
 * Later (T5-11b), a skill grant runs the tool loop: `test_all_grants_every_supplied_skill`,
 * `test_skills_with_structured_output_still_run_the_tool_loop` and the model-request half of
 * `test_memory_context_gets_the_system_prompt_without_the_listing`. Later (T5-11b, T6-7), they
 * also load files from a skill directory: `test_granted_skills_are_listed_and_loaded_on_demand`
 * and `test_escape_attempt_fails_the_node`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { Skill, SkillError } from "../../src/skills/catalog.ts";
import { LOAD_SKILL } from "../../src/skills/tool.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { drive, rejected, serving, textReply } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/** The failure message of an evaluation outcome; fails the test when it isn't a failure. */
function failureMessage(outcome: unknown): string {
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure.message;
}

const SKILLS: ReadonlyMap<string, Skill> = new Map([
  ["review", new Skill({ name: "review", description: "Review code changes.", instructions: "Read first." })],
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
  // A granted skill needs the tool loop, so the node stops after the memory context opens.
  const models = serving(new ScriptedModel([], CONFIG));
  const error = await rejected(node.forward({}, { openMemoryContext, skills: SKILLS, models }));
  expect((error as Error).message).toMatch(/tool loop is not ported yet/);
  expect(seen).toEqual([DEFAULT_SYSTEM]);
});

test("load_skill is not callable without a grant", async () => {
  const call = new ToolCallPart({ id: "1", name: LOAD_SKILL, arguments: { name: "review" } });
  const reply = new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "stop" });
  const outcome = await drive(new AgentNode("agent", { prompt: "p", llmConfig: CONFIG }), {
    models: serving(new ScriptedModel([reply], CONFIG)),
  });
  expect(failureMessage(outcome)).toContain("requested tool 'load_skill', but it is not granted");
});
