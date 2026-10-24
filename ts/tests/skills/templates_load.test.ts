/**
 * Every `composing-agents` skill template loads: a port of `tests/engine/test_templates_load.py`.
 * The templates are part of the authoring contract, so one that no longer loads is a bug.
 *
 * Changed: Python skips the templates written with TS yian's `ask`, `sleep` and `take`, list
 * comprehensions, `len()` and slices; TS loads them, and skips the two whose CODE bodies are
 * Python (`PYTHON_ONLY`).
 */

import { expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";

const TEMPLATES = fileURLToPath(new URL("../../../.claude/skills/composing-agents/templates", import.meta.url));
// Their CODE bodies are Python (a `module:function` reference, a Python lambda body).
const PYTHON_ONLY: ReadonlySet<string> = new Set(["inline-code.yaml", "pipeline.yaml"]);
const NAMES = readdirSync(TEMPLATES)
  .filter((name) => name.endsWith(".yaml") && !PYTHON_ONLY.has(name))
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
    "loop.yaml",
    "sleep-until.yaml",
  ]) {
    expect(NAMES).toContain(name);
  }
  for (const name of PYTHON_ONLY) expect(existsSync(join(TEMPLATES, name))).toBe(true);
});
