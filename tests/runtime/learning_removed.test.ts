/**
 * Learning is gone: a run result has no backward pass, and the removed options are refused by name.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const FIXTURES = fileURLToPath(new URL("../conformance/fixtures", import.meta.url));

const PLAN_FLOW = `
id: run-plan
name: plan_flow
input:
  topic: !str
nodes:
  plan:
    kind: code
    code: ./compose_codefns.ts:make_plan
    input:
      topic: \${input.topic}
    output:
      rating: str
      score: float
output:
  rating: \${plan.output.rating}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-learning-removed-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function load(): Promise<ComposedFlow> {
  return loadFlow(PLAN_FLOW, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** Run the one-CODE-node flow to a durable log in the test's folder. */
async function runSimple() {
  const log = join(dir, "run.jsonl");
  const result = await runFlow(await load(), { topic: "owls" }, { log, env: hostEnv(dir) });
  return { result, log };
}

test("a run result has no backward pass", async () => {
  const { result } = await runSimple();
  expect(result.status).toBe("succeeded");
  expect("backward" in result).toBe(false);
  expect("learning" in result).toBe(false);
  expect("backwardOutcome" in result).toBe(false);
});

test("runFlow refuses the removed learnable option by name", async () => {
  const flow = await load();
  await expect(runFlow(flow, { topic: "owls" }, { learnable: true, env: hostEnv(dir) } as never)).rejects.toThrow(
    "learnable was removed: yian no longer learns from runs; drop the option",
  );
});
