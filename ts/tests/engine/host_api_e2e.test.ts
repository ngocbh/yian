/**
 * Inbox runs end to end through the host calls: a run waits on a take of its `Inbox` input, the
 * host sends and closes, and `advance` moves it on. TS-only: Python yian has no inboxes.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { START_ID } from "../../src/compile/model.ts";
import { advance, close, runFlow, send } from "../../src/run.ts";
import { PoolCommitRecord, readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { rejected } from "./run_state_fixtures.ts";
import { processChatFlow, processEnv as env } from "./yian-host-api-child-hold.ts";

const ROOT = mkdtempSync(join(tmpdir(), "host-api-e2e-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

/** A `gate` tool that opens at once. */
const tools = {
  gate: new FunctionTool({
    name: "gate",
    description: "opens at once",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "opened",
  }),
};

test("an inbox input: START holds its handle, the run-start commit omits it; send, close, advance", async () => {
  const log = freshLog();
  const flow = processChatFlow();
  expect((await runFlow(flow, {}, { log, env, tools })).status).toBe("waiting");
  const records = readRunLog(log).map((envelope) => envelope.record);
  expect((records[0] as RunStartedRecord).startWiringOverride).toEqual({ m: { inbox: "m" } });
  const starts = records.filter(
    (record): record is PoolCommitRecord => record instanceof PoolCommitRecord && record.key === START_ID,
  );
  // The run start's input commit, then the START node's output commit.
  expect(starts.map((record) => record.value)).toEqual([
    { kind: "object", value: {} },
    { kind: "object", value: { m: { inbox: "m" } } },
  ]);
  await send(log, "m", "hi", { env, flow });
  await close(log, "m", { env, flow });
  const advanced = await advance({ log, flow, env, tools });
  expect([advanced.status, advanced.output]).toEqual(["succeeded", { events: ["hi"], closed: true }]);
});

test("a run input for an inbox is refused", async () => {
  const error = await rejected(runFlow(processChatFlow(), { m: ["hi"] }, { log: freshLog(), env, tools }));
  expect(error).toEqual(new RangeError("inbox `m` receives events through send(); it takes no run input"));
});
