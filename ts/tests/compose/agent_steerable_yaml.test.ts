/**
 * `steerable:` on AGENT: mirrors `tests/compose/test_agent_steerable_yaml.py`.
 *
 * Ported: `test_steerable_reaches_the_agent_node_and_defaults_off`,
 * `test_steerable_agent_without_declared_output_is_text`,
 * `test_bad_steerable_points_at_the_steerable_line` (loaded with `assembleText` where Python calls
 * `load_flow`), `test_agent_node_rejects_a_non_bool_steerable`, `test_needs_steering_follows_steerable`
 * and `test_steerable_is_identity_but_false_is_not` (hand-built nodes where Python loads `_flow`).
 *
 * TS-only: the harvested cases of these errors, as `parseNodes` or `assemble` reads them.
 */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { forget } from "../../src/defaults.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { assembled, assembleText, type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

/** A one-agent flow with `steerableLine` on line 7 and `output` after it. */
function steerFlow(steerableLine: string, output = "    output: str"): string {
  return `\
id: steer
name: steer
nodes:
  chat:
    kind: agent
    prompt: Help the user.
${steerableLine}
${output}
output: \${chat.output}
`;
}

/** Assemble `text` and return its `chat` node. */
const chatOf = (text: string) => assembleText(text).flow.compiled.nodes.get("chat") as AgentNode;

test("steerable reaches the agent node and defaults off", () => {
  expect(chatOf(steerFlow("    steerable: true")).steerable).toBe(true);
  expect(chatOf(steerFlow("")).steerable).toBe(false);
});

test("steerable agent without declared output is text", () => {
  expect(chatOf(steerFlow("    steerable: true", "")).steerable).toBe(true);
});

test.each([
  ["    steerable: yes please", "    output: str", "`steerable:` must be true or false"],
  ["    steerable: yes", "    output: str", "`steerable:` must be true or false; `yes` is text; write `true`"],
  ["    steerable: true", "    output: int", "steerable needs `str` output"],
  ["    steerable: true", "    output: {answer: str}", "steerable needs `str` output"],
])("bad steerable points at the steerable line: %p %p", (line, output, message) => {
  const error = loadError(() => assembleText(steerFlow(line, output)));

  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(7);
});

test("agent node rejects a non-bool steerable", () => {
  expect(() => new AgentNode("chat", { prompt: "p", steerable: 1 as unknown as boolean })).toThrow(
    new TypeError("AGENT node 'chat': steerable must be a bool"),
  );
  expect(() => new AgentNode("chat", { prompt: "p", steerable: 1 as unknown as boolean })).toThrow(TypeError);
});

test("needsSteering follows steerable", () => {
  expect(new AgentNode("chat", { prompt: "p", steerable: true }).needsSteering).toBe(true);
  expect(new AgentNode("chat", { prompt: "p" }).needsSteering).toBe(false);
});

test("steerable is identity but false is not", () => {
  const fingerprint = (agent: AgentNode) =>
    flowFingerprint(
      new Flow(new Map([[agent.id, agent]]), [], {
        wiring: new Map([[agent.id, new Map()]]),
        startId: agent.id,
        endId: agent.id,
      }),
    );
  const base = fingerprint(new AgentNode("chat", { prompt: "Help the user." }));

  expect(fingerprint(new AgentNode("chat", { prompt: "Help the user.", steerable: false }))).toBe(base);
  expect(fingerprint(new AgentNode("chat", { prompt: "Help the user.", steerable: true }))).not.toBe(base);
});

test.each<CaseRow>([
  { id: "load-agent/compose_agent_steerable_yaml__steerable_not_a_bool", call: parsedNodes },
  { id: "load-agent/compose_agent_steerable_yaml__steerable_yes_is_text", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  { id: "load-agent/compose_agent_steerable_yaml__steerable_needs_str_output", call: assembled },
])("assemble: $id", runCase);
