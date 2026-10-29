/** `withAppendHook`: hooks on `JsonlRunLog.prototype.append` scoped to a call's async work. */

import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { readRunLog } from "../../src/runtime/log.ts";
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

/** A log path in a fresh dir that is removed after the test. */
function freshLog(): string {
  const dir = mkdtempSync(join(tmpdir(), "append-hook-"));
  made.push(dir);
  return join(dir, "run.jsonl");
}

/** Run `FLOW` to `log`, a fresh log by default; return the log's path. */
async function run(log = freshLog()): Promise<string> {
  await runFlow(await loadFlow(FLOW), {}, { log, env: NO_ENV });
  return log;
}

test("a hook sees every append its body makes, after awaits too, and no other run's", async () => {
  const seen: (string | null)[] = [];
  const inside = withAppendHook((log) => seen.push(log.path), run);
  // Started outside the hook's scope while the hooked run goes on.
  const outside = run();
  const [hooked, other] = await Promise.all([inside, outside]);
  expect(seen.length).toBeGreaterThan(1);
  expect(new Set(seen)).toEqual(new Set([hooked]));
  expect(seen).not.toContain(other);
});

test("a hook sees an in-memory log's appends too", async () => {
  const seen: (string | null)[] = [];
  const result = await withAppendHook(
    (log) => seen.push(log.path),
    async () => runFlow(await loadFlow(FLOW), {}, { log: null, env: NO_ENV }),
  );
  expect(result.status).toBe("succeeded");
  expect(seen.length).toBeGreaterThan(1);
  expect(new Set(seen)).toEqual(new Set([null]));
});

test("a hook that throws refuses the append", async () => {
  const refused = new Error("refused");
  const log = freshLog();
  let calls = 0;
  const crashed = withAppendHook(
    () => {
      calls += 1;
      if (calls === 3) throw refused;
    },
    () => run(log),
  );
  await expect(crashed).rejects.toBe(refused);
  // Only the two appends before the refused one reached the log.
  expect(readRunLog(log)).toHaveLength(2);
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
