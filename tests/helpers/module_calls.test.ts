/** `withModuleCalls`: hooks on the calls sent to tool modules' threads, scoped to a call's async work. */

import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { withModuleCalls } from "./module_calls.ts";
import { NO_ENV } from "./run_state_fixtures.ts";

const made: string[] = [];
afterEach(() => {
  for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
});

const FLOW = `id: price
name: price
uses:
  tools: ./tools.ts
nodes:
  priced:
    kind: call
    call: {tools: quote}
    input:
      qty: 3
output: \${priced.output}
`;

const MODULE = `export const quote = {
  name: "quote",
  description: "Price an order.",
  parameters: { type: "object", properties: { qty: { type: "integer" } }, required: ["qty"] },
  execute: (args) => args.qty * 2,
};
`;

/** Run `FLOW` in a fresh dir with its module, under run id `runId`. */
async function run(runId: string): Promise<unknown> {
  const dir = mkdtempSync(join(tmpdir(), "module-calls-"));
  made.push(dir);
  writeFileSync(join(dir, "tools.ts"), MODULE);
  const result = await runFlow(await loadFlow(FLOW, { baseDir: dir }), {}, { runId, env: NO_ENV });
  return result.output;
}

test("a hook sees each module call its body sends, with the module's key and export, and no other run's", async () => {
  const seen: [string, string, string][] = [];
  const inside = withModuleCalls((ctx, module, tool) => seen.push([ctx.runId!, module, tool]), () => run("inside"));
  // Started outside the hook's scope while the hooked run goes on.
  const outside = run("outside");
  expect(await Promise.all([inside, outside])).toEqual([6n, 6n]);
  expect(seen).toEqual([["inside", "tools.ts", "quote"]]);
});
