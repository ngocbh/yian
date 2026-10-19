/**
 * Retired native settings fail before a flow can perform external work: mirrors
 * `tests/engine/test_provider_retirement.py`.
 *
 * Ported: `test_retired_tool_marker_is_not_an_implicit_tool_grant` and
 * `test_old_native_yaml_fails_during_loading` (the root flow assembled as the loader assembles
 * it), and the harvested case `engine_provider_retirement__provider_codex_is_rejected`.
 *
 * Elsewhere: `test_retired_native_config_points_to_unified_auth` and
 * `test_codex_is_not_a_separate_provider_route` belong to the `llm_clients` ports.
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { assembled, assembleText, loadError, runCase } from "../compose/declarations.ts";

test("retired tool marker is not an implicit tool grant", () => {
  expect(() => new AgentNode("agent", { tools: ["builtin_tools"] })).toThrow(RangeError);
  expect(() => new AgentNode("agent", { tools: ["builtin_tools"] })).toThrow(
    "grant registered Yian tools explicitly",
  );
});

test("old native yaml fails during loading", () => {
  const text = `id: old
name: old
nodes:
  agent:
    kind: agent
    prompt: hi
    llm_config: {provider: codex, model: old-model}
output: \${agent.output}
`;
  expect(loadError(() => assembleText(text)).message).toContain("yian auth openai");
});

test.each(["load-agent/engine_provider_retirement__provider_codex_is_rejected"])("case %s", (id) =>
  runCase({ id, call: assembled }),
);
