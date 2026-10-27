/** `system_prompt:` on AGENT. */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
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

/** Load `text` and return its compiled flow. */
const compiledOf = async (text: string) => (await loadFlow(text)).compiled;

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

test("flow without system prompt keeps its fingerprint", async () => {
  // An absent `system_prompt:` adds nothing to identity. Pinned after `mode` left the AGENT
  // manifest.
  for (const compiled of [await compiledOf(chatFlow("")), await compiledOf(compactFlow(""))]) {
    expect("system_prompt" in ((flowManifest(compiled).nodes as JsonObject).reply as JsonObject)).toBe(false);
  }
  expect(flowFingerprint(await compiledOf(chatFlow("")))).toBe(
    "fc15ce95efcac79320ef54dad98e219283ab094c8521dbdcd8d6a9f65397e801",
  );
  expect(flowFingerprint(await compiledOf(compactFlow("")))).toBe(
    "a86275835f7f45b1a3eb6f9f4604cec49955a760339c13f0de26d9a539c47312",
  );
});

test("system prompt is identity", async () => {
  const base = flowFingerprint(await compiledOf(chatFlow("")));
  const persona = flowFingerprint(await compiledOf(chatFlow(`    system_prompt: "${PERSONA}"`)));
  const other = flowFingerprint(await compiledOf(chatFlow('    system_prompt: "You are ${name}."')));

  expect(new Set([base, persona, other]).size).toBe(3);
});

test.each<CaseRow>([
  { id: "load-agent/compose_agent_system_prompt_yaml__int_system_prompt", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  { id: "load-agent/compose_agent_system_prompt_yaml__compact_flow_accepts_system_prompt", call: assembled },
  { id: "load-agent/compose_agent_system_prompt_yaml__pool_ref_in_system_prompt", call: assembled },
])("assembles: $id", runCase);
