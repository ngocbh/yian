/**
 * `system_prompt:` on AGENT: mirrors `tests/compose/test_agent_system_prompt_yaml.py`.
 *
 * Ported: `test_agent_node_rejects_a_non_string_system_prompt`, `test_system_prompt_defaults_to_none`,
 * `test_system_prompt_reaches_the_agent_node`, `test_compact_flow_accepts_system_prompt`,
 * `test_system_prompt_sees_the_node_variables`, `test_unknown_ref_points_at_the_system_prompt_line`,
 * `test_unknown_ref_in_a_compact_flow_is_located`, `test_non_string_system_prompt_points_at_its_line`
 * (loaded with `assembleText` where Python calls `load_flow`), the manifest-key half of
 * `test_flow_without_system_prompt_keeps_its_fingerprint` and `test_system_prompt_is_identity`.
 *
 * Changed: the message says `or null` where Python's says `or None`. The identity rows build the
 * `reply` node by hand where Python loads `_flow`.
 *
 * TS-only: the harvested cases of the `123` row, as `parseNodes` reads it, and of the compact
 * flow and the pool ref, which assemble.
 *
 * Later (T3-22): the pinned fingerprints of
 * `test_flow_without_system_prompt_keeps_its_fingerprint` (loaded `_flow` and `_compact`).
 */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { forget } from "../../src/defaults.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { assembled, assembleText, type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

const PERSONA = "You are ${name}, ${title}. Speak as ${name} in the first person.";

/** A flow whose `reply` agent has `systemLine` on line 13. */
function chatFlow(systemLine: string): string {
  return `\
id: chat
name: chat
input:
  name: !str
  title: !str
  message: !str
nodes:
  reply:
    kind: agent
    input: {name: "\${input.name}", title: "\${input.title}", message: "\${input.message}"}
    variables:
      tone: !str warm
${systemLine}
    prompt: \${message}
    output: str
output: \${reply.output}
`;
}

/** A compact flow, its one agent named `reply`, with `systemLine` on line 9. */
function compactFlow(systemLine: string): string {
  return `\
id: reply
name: reply
input:
  name: !str
  title: !str
  message: !str
output: str
kind: agent
${systemLine}
prompt: \${message}
`;
}

/** Assemble `text` and return its `reply` node. */
const replyOf = (text: string) => assembleText(text).flow.compiled.nodes.get("reply") as AgentNode;

/** A one-node flow of the `reply` agent, with `systemPrompt` when given. */
function replyFlow(systemPrompt: string | null = null): Flow {
  const reply = new AgentNode("reply", { prompt: "${message}", systemPrompt });
  return new Flow(new Map([["reply", reply]]), [], {
    wiring: new Map([["reply", new Map()]]),
    startId: "reply",
    endId: "reply",
  });
}

test("system prompt reaches the agent node", () => {
  expect(replyOf(chatFlow(`    system_prompt: "${PERSONA}"`)).systemPrompt).toBe(PERSONA);
});

test("system prompt defaults to null", () => {
  expect(replyOf(chatFlow("")).systemPrompt).toBeNull();
  expect(new AgentNode("reply").systemPrompt).toBeNull();
});

test("compact flow accepts system prompt", () => {
  expect(replyOf(compactFlow(`system_prompt: "${PERSONA}"`)).systemPrompt).toBe(PERSONA);
});

test("system prompt sees the node variables", () => {
  expect(replyOf(chatFlow('    system_prompt: "Be ${tone}."')).systemPrompt).toBe("Be ${tone}.");
});

test.each([
  ["${nickname}", ""],
  ["${input.name}", "a pool ref belongs in an input `from:`"],
])("unknown ref %p points at the system prompt line", (ref, hint) => {
  const error = loadError(() => assembleText(chatFlow(`    system_prompt: "You are ${ref}."`)));

  expect(error.message).toMatch(/system_prompt: .* is not a declared input or variable/);
  expect(error.message).toContain(hint);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(13);
});

test("unknown ref in a compact flow is located", () => {
  const error = loadError(() => assembleText(compactFlow('system_prompt: "You are ${nickname}."')));

  expect(error.message).toMatch(/system_prompt: .* is not a declared input/);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(9);
});

test.each(["123", "[a, b]", "{a: b}", "null", '""'])("non-string system prompt %p points at its line", (value) => {
  const error = loadError(() => assembleText(chatFlow(`    system_prompt: ${value}`)));

  expect(error.message).toContain("`system_prompt:` must be a non-empty string");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(13);
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

test.each<CaseRow>([
  { id: "load-agent/compose_agent_system_prompt_yaml__compact_flow_accepts_system_prompt", call: assembled },
  { id: "load-agent/compose_agent_system_prompt_yaml__pool_ref_in_system_prompt", call: assembled },
])("assembles: $id", runCase);
