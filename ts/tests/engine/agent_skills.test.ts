/**
 * AGENT skill grants: mirrors `tests/engine/test_agent_skills.py`.
 *
 * Ported: `test_agent_rejects_invalid_skill_grants` (5 rows) and
 * `test_agent_skill_grants_default_to_none`; the `grantedSkills` halves of
 * `test_missing_grant_fails_the_node` and `test_grant_without_a_skills_mapping_fails_the_node`.
 *
 * Ported (adapted): `test_only_nodes_with_the_skills_trait_receive_skills`, on a local probe that
 * records its capabilities where Python's `FuncNode` would refuse an unexpected keyword: a node
 * without `needsSkills` gets no `skills`, and one with it gets the host's map itself.
 *
 * Changed: the grant checks are called directly where Python runs the node through `eval_node`;
 * a list of names in a message is written in the text form (`["absent"]`, Python
 * `['absent']`).
 *
 * TS-only: `grantedSkills` returns no skills without a grant, and the granted ones sorted by
 * name.
 *
 * Later (T5-11a), they run the node: the node-failure halves of
 * `test_missing_grant_fails_the_node` and `test_grant_without_a_skills_mapping_fails_the_node`,
 * `test_all_grants_every_supplied_skill`,
 * `test_no_grant_or_no_supply_keeps_the_system_prompt_byte_for_byte`,
 * `test_memory_context_gets_the_system_prompt_without_the_listing`,
 * `test_granted_skills_are_listed_and_loaded_on_demand`,
 * `test_skills_with_structured_output_still_run_the_tool_loop`,
 * `test_escape_attempt_fails_the_node` and `test_load_skill_is_not_callable_without_a_grant`
 * (T5-11a, T6-7).
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { Skill, SkillError } from "../../src/skills/catalog.ts";
import { LOAD_SKILL } from "../../src/skills/tool.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

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
