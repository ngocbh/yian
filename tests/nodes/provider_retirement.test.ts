/** Retired native settings fail before a flow can perform external work. */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { assembled, assembleText, loadError, runCase } from "../helpers/declarations.ts";

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
