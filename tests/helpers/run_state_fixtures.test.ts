/** `crashAfter` and `crashes`: simulated crashes scoped to the work they run. */

import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { crashAfter, CrashAt, crashes, NO_ENV } from "./run_state_fixtures.ts";

const made: string[] = [];
afterEach(() => {
  for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
});

const FLOW = `id: two
name: two
nodes:
  a: {kind: code, code: return "x", output: str}
output: \${a.output}
`;

/** Run `FLOW` to a fresh log; return its status. */
async function run(): Promise<string> {
  const dir = mkdtempSync(join(tmpdir(), "crash-scope-"));
  made.push(dir);
  const result = await runFlow(await loadFlow(FLOW), {}, { log: join(dir, "run.jsonl"), env: NO_ENV });
  return result.status;
}

/** A promise and the function that resolves it. */
function gate(): [Promise<void>, () => void] {
  let open!: () => void;
  const opened = new Promise<void>((resolve) => (open = resolve));
  return [opened, open];
}

// A test that times out leaves its crash helper running; another test's run must not crash.
test("crashAfter crashes only the runs it starts, not one started beside it", async () => {
  const [opened, open] = gate();
  const crashed = crashAfter(1, async () => {
    await opened;
    await run();
  });
  expect(await run()).toBe("succeeded");
  open();
  await crashed;
});

test("crashes crashes only the runs it starts, not one started beside it", async () => {
  const model = new CrashAt([], 1);
  model.frozen = true;
  const [opened, open] = gate();
  const crashed = crashes(model, async () => {
    await opened;
    await run();
  });
  expect(await run()).toBe("succeeded");
  open();
  await crashed;
});
