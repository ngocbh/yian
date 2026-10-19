/**
 * `system_prompt:` on AGENT: mirrors `tests/compose/test_agent_system_prompt_yaml.py`.
 *
 * Ported: `test_agent_node_rejects_a_non_string_system_prompt`, the constructor half of
 * `test_system_prompt_defaults_to_none`, the manifest-key half of
 * `test_flow_without_system_prompt_keeps_its_fingerprint` and `test_system_prompt_is_identity`.
 *
 * Changed: the message says `or null` where Python's says `or None`. The identity rows build the
 * `reply` node by hand where Python loads `_flow`.
 *
 * The `123` row of `test_non_string_system_prompt_points_at_its_line` is pinned through its
 * harvested case, as `parseNodes` reads it.
 *
 * Later (T3-15): the `load_flow` half of `test_system_prompt_defaults_to_none`,
 * `test_system_prompt_reaches_the_agent_node`, `test_compact_flow_accepts_system_prompt`,
 * `test_system_prompt_sees_the_node_variables`, `test_unknown_ref_points_at_the_system_prompt_line`,
 * `test_unknown_ref_in_a_compact_flow_is_located` and the other rows of
 * `test_non_string_system_prompt_points_at_its_line`, and the pinned fingerprints of
 * `test_flow_without_system_prompt_keeps_its_fingerprint` (loaded `_flow` and `_compact`).
 */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { forget } from "../../src/defaults.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { type CaseRow, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

const PERSONA = "You are ${name}, ${title}. Speak as ${name} in the first person.";

/** A one-node flow of the `reply` agent, with `systemPrompt` when given. */
function replyFlow(systemPrompt: string | null = null): Flow {
  const reply = new AgentNode("reply", { prompt: "${message}", systemPrompt });
  return new Flow(new Map([["reply", reply]]), [], {
    wiring: new Map([["reply", new Map()]]),
    startId: "reply",
    endId: "reply",
  });
}

test("system prompt defaults to null", () => {
  expect(new AgentNode("reply").systemPrompt).toBeNull();
});

test("agent node rejects a non-string system prompt", () => {
  const message = "AGENT node 'reply': system_prompt must be a non-empty string or null";
  expect(() => new AgentNode("reply", { systemPrompt: 7 as unknown as string })).toThrow(new TypeError(message));
  expect(() => new AgentNode("reply", { systemPrompt: 7 as unknown as string })).toThrow(TypeError);
  expect(() => new AgentNode("reply", { systemPrompt: "" })).toThrow(new TypeError(message));
  expect(() => new AgentNode("reply", { systemPrompt: "" })).toThrow(TypeError);
});

test("flow without system prompt keeps its fingerprint (the manifest key)", () => {
  expect("system_prompt" in ((flowManifest(replyFlow()).nodes as JsonObject).reply as JsonObject)).toBe(false);
});

test("system prompt is identity", () => {
  const base = flowFingerprint(replyFlow());
  const persona = flowFingerprint(replyFlow(PERSONA));
  const other = flowFingerprint(replyFlow("You are ${name}."));

  expect(new Set([base, persona, other]).size).toBe(3);
});

test.each<CaseRow>([
  { id: "load-agent/compose_agent_system_prompt_yaml__int_system_prompt", call: parsedNodes },
])("parseNodes: $id", runCase);
