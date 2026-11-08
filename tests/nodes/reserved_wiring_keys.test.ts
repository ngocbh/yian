/**
 * `Node.reservedWiringKeys()`, the static author-wiring keys a node owns beyond its declared
 * `params` (MAP `over`).
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";

test("reserved wiring keys by kind", () => {
  expect(new MapNode("m", { flowId: "c", child: null, childInputs: [] }).reservedWiringKeys()).toEqual(
    new Set(["over"]),
  );
  // A plain leaf reserves nothing.
  expect(new CodeNode("c", { code: "./mod.ts:fn" }).reservedWiringKeys()).toEqual(new Set());
  // An AGENT reserves `__llm_config` only once Compose wires values in its layers.
  const agent = new AgentNode("a", { prompt: "p" });
  expect(agent.reservedWiringKeys()).toEqual(new Set());
  agent.wiresLlmConfig = true;
  expect(agent.reservedWiringKeys()).toEqual(new Set(["__llm_config"]));
});

test("an AGENT reserves `__memory_<input>` for each memory its grants name, once, and a TOOL node for its memory's tool", () => {
  const tools = ["lessons", { kind: "group", group: "lessons", tools: ["read"] }, "search"] as const;
  const agent = new AgentNode("a", { prompt: "p", tools: [...tools], memoryInputs: ["lessons"] });
  expect(agent.reservedWiringKeys()).toEqual(new Set(["__memory_lessons"]));
  expect(agent.reservedKeyLabel("__memory_lessons")).toBe("tools");
  const tool = new ToolNode("t", { toolId: { group: "lessons", name: "read" }, memoryTool: true });
  expect(tool.reservedWiringKeys()).toEqual(new Set(["__memory_lessons"]));
  expect(tool.reservedKeyLabel("__memory_lessons")).toBe("tool_id");
  expect(new ToolNode("h", { toolId: { group: "github", name: "search" } }).reservedWiringKeys()).toEqual(new Set());
});
