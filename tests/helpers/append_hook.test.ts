/** `withAppendHook`: hooks on `JsonlRunLog.prototype.append` scoped to a call's async work. */

import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import type { JsonlRunLog } from "../../src/runtime/log.ts";
import { withAppendHook } from "./append_hook.ts";
import { NO_ENV } from "./run_state_fixtures.ts";

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

/** Run `FLOW` to a fresh log; return the log's path. */
async function run(): Promise<string> {
  const dir = mkdtempSync(join(tmpdir(), "append-hook-"));
  made.push(dir);
  const log = join(dir, "run.jsonl");
  await runFlow(await loadFlow(FLOW), {}, { log, env: NO_ENV });
  return log;
}

test("a hook sees every append its body makes, after awaits too, and no other run's", async () => {
  const seen: string[] = [];
  const inside = withAppendHook((log: JsonlRunLog) => seen.push(log.path), run);
  // Started outside the hook's scope while the hooked run goes on.
  const outside = run();
  const [hooked, other] = await Promise.all([inside, outside]);
  expect(seen.length).toBeGreaterThan(1);
  expect(new Set(seen)).toEqual(new Set([hooked]));
  expect(seen).not.toContain(other);
});

test("a hook that throws refuses the append", async () => {
  const refused = new Error("refused");
  const crashed = withAppendHook((log) => {
    if (log.nextSequence > 2) throw refused;
  }, run);
  await expect(crashed).rejects.toBe(refused);
});

test("a nested hook runs after the enclosing one", async () => {
  const order: string[] = [];
  await withAppendHook(
    () => order.push("outer"),
    () => withAppendHook(() => order.push("inner"), run),
  );
  expect(order.length).toBeGreaterThan(0);
  expect(order.slice(0, 2)).toEqual(["outer", "inner"]);
  expect(order.filter((name) => name === "outer")).toHaveLength(order.length / 2);
});
