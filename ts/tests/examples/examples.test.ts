/**
 * Every shipped example loads and compiles (no live LLM): mirrors
 * `tests/examples/test_examples.py`.
 *
 * Ported: `test_example_loads_and_compiles` (one test per `examples/*.yaml`, loaded with the
 * examples directory as the search path), `test_examples_present` and
 * `test_debate_agents_are_tool_free`.
 */

import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";

const EXAMPLES_DIR = fileURLToPath(new URL("../../../examples", import.meta.url));
const EXAMPLES = readdirSync(EXAMPLES_DIR)
  .filter((name) => name.endsWith(".yaml"))
  .sort();

const load = (name: string) =>
  loadFlow(readFileSync(join(EXAMPLES_DIR, name), "utf8"), { searchPaths: [EXAMPLES_DIR] });

test.each(EXAMPLES)("example %s loads and compiles", async (name) => {
  const loaded = await load(name);
  expect(loaded.compiled).toBeDefined();
  expect(loaded.input).toBeDefined();
});

test("examples present", () => {
  for (const name of ["hello.yaml", "summarize.yaml", "classify.yaml"]) {
    expect(EXAMPLES).toContain(name);
  }
});

test("debate agents are tool-free", async () => {
  const loaded = await load("debate.yaml");
  for (const nodeId of ["frame", "for_case", "against_case", "verdict"]) {
    const node = loaded.compiled.nodes.get(nodeId) as AgentNode;
    expect(node.tools).toEqual([]);
  }
});
