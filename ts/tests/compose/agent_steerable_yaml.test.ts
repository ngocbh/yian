/**
 * `steerable:` on AGENT: mirrors `tests/compose/test_agent_steerable_yaml.py`.
 *
 * Ported: `test_agent_node_rejects_a_non_bool_steerable`, `test_needs_steering_follows_steerable`
 * and `test_steerable_is_identity_but_false_is_not` (hand-built nodes where Python loads `_flow`).
 *
 * The parse-time rows of `test_bad_steerable_points_at_the_steerable_line` (a non-bool
 * `steerable:`) are pinned through their harvested cases, as `parseNodes` reads them.
 *
 * Later (T3-15): `test_steerable_reaches_the_agent_node_and_defaults_off`,
 * `test_steerable_agent_without_declared_output_is_text` and the `steerable needs \`str\` output`
 * rows of `test_bad_steerable_points_at_the_steerable_line`.
 */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { forget } from "../../src/defaults.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { type CaseRow, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

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
