/**
 * Every `composing-agents` skill template loads: a port of `tests/engine/test_templates_load.py`.
 * The templates are part of the authoring contract, so one that no longer loads is a bug.
 */

import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";

const TEMPLATES = fileURLToPath(new URL("../../../.claude/skills/composing-agents/templates", import.meta.url));
const NAMES = readdirSync(TEMPLATES)
  .filter((name) => name.endsWith(".yaml"))
  .sort();

test.each(NAMES)("template %s loads", async (name) => {
  const flow = await loadFlow(readFileSync(join(TEMPLATES, name), "utf8"), { searchPaths: [TEMPLATES] });
  expect(flow.compiled).toBeDefined();
});

test("the migrated templates are among those loaded", () => {
  for (const name of [
    "chat.yaml",
    "compact_history.yaml",
    "human-in-loop.yaml",
    "human-questions.yaml",
    "inline-code.yaml",
    "loop.yaml",
    "pipeline.yaml",
    "sleep-until.yaml",
  ]) {
    expect(NAMES).toContain(name);
  }
});
