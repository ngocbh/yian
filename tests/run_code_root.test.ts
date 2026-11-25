/**
 * A run of a flow loaded under `codeRoot` holds what its code imports to that root: a CODE step
 * whose module imports a file outside fails, naming the import, at the start and after an
 * `advance` alike, and a tool module that does refuses the run's start. A flow loaded without one
 * runs as before.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import { advance, complete, runFlow } from "../src/run.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import { codeRootTree } from "./helpers/code_root_routes.ts";
import { NO_ENV, rejected } from "./helpers/run_state_fixtures.ts";

const tree = codeRootTree();
afterAll(() => rmSync(tree.base, { recursive: true, force: true }));

const REFUSED = "imports '../outside/evil.ts', which resolves outside the code root";

/** Write `files` in the root and load `flow.yaml`'s text `text` from it, under `codeRoot`. */
async function loaded(text: string, files: Record<string, string>, codeRoot: string | null) {
  for (const [name, source] of Object.entries(files)) writeFileSync(join(tree.root, name), source);
  return await loadFlow(text, { searchPaths: [tree.root], sourceLabel: "flow.yaml", codeRoot });
}

const CODE_FLOW = `id: confined
name: confined
nodes:
  read:
    kind: code
    code: ./read.ts:read
    output: !str
output: \${read.output}
`;

const READ = 'import { x } from "../outside/evil.ts";\nexport function read() { return x; }\n';

const MODULE_FLOW = `id: confined
name: confined
uses:
  crm: ./crm.ts
nodes:
  find:
    kind: call
    call: {crm: find}
    output: !str
output: \${find.output}
`;

const CRM = 'import { x } from "../outside/evil.ts";\nexport const find = { name: "find", description: "d", parameters: { type: "object" }, execute: () => x };\n';

test("a CODE step of a run under codeRoot fails when its module imports a file outside", async () => {
  const result = await runFlow(await loaded(CODE_FLOW, { "read.ts": READ }, tree.root), {}, { env: NO_ENV });
  expect(result.status).toBe("failed");
  expect(result.failure?.message).toContain(`read.ts ${REFUSED}`);
});

test("a run under codeRoot refuses to start when a tool module imports a file outside", async () => {
  const flow = await loaded(MODULE_FLOW, { "crm.ts": CRM }, tree.root);
  expect(((await rejected(runFlow(flow, {}, { env: NO_ENV }))) as Error).message).toContain(`crm.ts ${REFUSED}`);
});

test("a run without codeRoot loads the files its code imports from outside, as before", async () => {
  const code = await runFlow(await loaded(CODE_FLOW, { "read.ts": READ }, null), {}, { env: NO_ENV });
  const module = await runFlow(await loaded(MODULE_FLOW, { "crm.ts": CRM }, null), {}, { env: NO_ENV });
  expect([code.output, module.output]).toEqual(["OUTSIDE", "OUTSIDE"]);
});

test("an advance holds the run's CODE steps to the code root it recorded", async () => {
  const text = `id: confined
name: confined
nodes:
  ask:
    kind: call
    call: lookup
    output: !str
  read:
    kind: code
    input: {after: "\${ask.output}"}
    code: ./later.ts:read
    output: !str
output: \${read.output}
`;
  const flow = await loaded(text, { "later.ts": READ.replace("read()", "read(inputs: { after: string })") }, tree.root);
  const log = join(mkdtempSync(join(tree.base, "runs-")), "run.jsonl");
  const lookup = new FunctionTool({
    name: "lookup",
    description: "answers later",
    parametersJsonSchema: { type: "object" },
    execute: () => pending(),
  });
  const first = await runFlow(flow, {}, { log, env: NO_ENV, tools: { lookup } });
  expect(first.status).toBe("waiting");
  await complete(log, `${first.runId}/ask`, { output: "yes" });

  const resumed = await advance({ log, env: NO_ENV, tools: { lookup } });

  expect(resumed.status).toBe("failed");
  expect(resumed.failure?.message).toContain(`later.ts ${REFUSED}`);
});
