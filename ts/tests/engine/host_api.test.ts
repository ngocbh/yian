/**
 * The host calls on a run log: `send`, `complete`, `close`, `cancel` and `pendingCalls`, live
 * through this process's engine or offline under the log's lock, with their typed refusals and
 * `RunBusyError`. TS-only: Python yian has no inboxes or pending calls.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { FlowOutput, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FileLockError, lockFile } from "../../src/platform.ts";
import { advance, RunBusyError, runFlow, send } from "../../src/run.ts";
import { PoolCommitRecord, readRunLog } from "../../src/runtime/log.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { graph, NO_ENV, rejected } from "./run_state_fixtures.ts";

const ROOT = mkdtempSync(join(tmpdir(), "host-api-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

const env = NO_ENV;

/**
 * START with an `Inbox(str)` input `m` -> a TOOL `take: all` from it, `heard` -> END, which
 * returns `heard`'s output; `extra` nodes hang off START and feed END too.
 */
function chatFlow(extra: readonly Node[] = []): ComposedFlow {
  const str = Type.scalar(ValueKind.STRING);
  const start = new StartNode(START_ID, { params: [new Parameter("m", { contract: new ValueContract(inboxType(str)) })] });
  const heard = new ToolNode("heard", { toolId: "take" });
  heard.params = [new Parameter("from"), new Parameter("take")];
  const wiring: Record<string, Record<string, unknown>> = { heard: { from: "${input.m}", take: "all" } };
  for (const node of extra) wiring[node.id] = {};
  return new ComposedFlow(
    graph(
      [start, heard, ...extra],
      [[START_ID, "heard"], ...extra.map((node) => [START_ID, node.id] as const)],
      { wiring, outputs: [new FlowOutput("heard", "${heard.output}")] },
    ),
  );
}

/** A waiting run of `chatFlow()` at a fresh log. */
async function waitingChat() {
  const log = freshLog();
  const flow = chatFlow();
  const result = await runFlow(flow, {}, { log, env });
  expect(result.status).toBe("waiting");
  return { log, flow, runId: result.runId };
}

test("send to an idle run appends one send; advance consumes it", async () => {
  const { log, flow } = await waitingChat();
  const seq = await send(log, "m", "hi", { env, flow });
  const record = readRunLog(log)[seq]!.record as PoolCommitRecord;
  expect(record).toBeInstanceOf(PoolCommitRecord);
  expect(record.key).toBe(`${START_ID}.m`);
  expect((await advance({ log, flow, env })).output).toEqual({ events: ["hi"], closed: false });
});

test("a send while another writer holds the lock is RunBusyError and appends nothing", async () => {
  const { log, flow } = await waitingChat();
  const before = readFileSync(log);
  const holder = lockFile(`${realpathSync(log)}.lock`);
  try {
    const error = (await rejected(send(log, "m", "hi", { env, flow }))) as RunBusyError;
    expect(error).toBeInstanceOf(RunBusyError);
    expect([error.log, error.cause instanceof FileLockError]).toEqual([resolve(log), true]);
    expect(error.message).toBe(`run log ${resolve(log)} is locked by another writer (is the run still active?)`);
  } finally {
    holder.release();
  }
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("advance while another writer holds the lock is RunBusyError with the lock error as cause", async () => {
  const { log, flow } = await waitingChat();
  const holder = lockFile(`${realpathSync(log)}.lock`);
  try {
    const error = (await rejected(advance({ log, flow, env }))) as RunBusyError;
    expect(error).toBeInstanceOf(RunBusyError);
    expect(error.cause).toBeInstanceOf(FileLockError);
  } finally {
    holder.release();
  }
});
