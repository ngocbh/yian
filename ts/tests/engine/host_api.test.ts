/**
 * The host calls on a run log: `send`, `complete`, `close`, `cancel` and `pendingCalls`, live
 * through this process's engine or offline under the log's lock, with their typed refusals and
 * `RunBusyError`. TS-only: Python yian has no inboxes or pending calls.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { FlowOutput, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FileLockError, lockFile } from "../../src/platform.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { DefaultsMismatchError } from "../../src/defaults.ts";
import {
  advance,
  cancel,
  close,
  complete,
  DeliveryRefused,
  type DeliveryRefusal,
  pendingCalls,
  RunBusyError,
  runFlow,
  send,
} from "../../src/run.ts";
import { PoolCommitRecord, readRunLog, RunCancelledRecord } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { inboxType, Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { handClock } from "./inbox_fixtures.ts";
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
 * returns `heard`'s output; `extra` nodes hang off START, wired from `extraWiring` by id.
 */
function chatFlow(extra: readonly Node[] = [], extraWiring: Record<string, Record<string, unknown>> = {}): ComposedFlow {
  const str = Type.scalar(ValueKind.STRING);
  const start = new StartNode(START_ID, { params: [new Parameter("m", { contract: new ValueContract(inboxType(str)) })] });
  const heard = new ToolNode("heard", { toolId: "take" });
  heard.params = [new Parameter("from"), new Parameter("take")];
  const wiring: Record<string, Record<string, unknown>> = { heard: { from: "${input.m}", take: "all" } };
  for (const node of extra) wiring[node.id] = extraWiring[node.id] ?? {};
  return new ComposedFlow(
    graph(
      [start, heard, ...extra],
      [[START_ID, "heard"], ...extra.map((node) => [START_ID, node.id] as const)],
      { wiring, outputs: [new FlowOutput("heard", "${heard.output}")] },
    ),
  );
}

/** A waiting run of `chatFlow()` at a fresh log, under `runEnv` (default `env`). */
async function waitingChat(runEnv: typeof env = env) {
  const log = freshLog();
  const flow = chatFlow();
  const result = await runFlow(flow, {}, { log, env: runEnv });
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

/** Expect `call` to reject with `DeliveryRefused` for `reason` and leave the file at `log` unchanged; return the error. */
async function refusedWith(log: string, call: () => Promise<unknown>, reason: DeliveryRefusal): Promise<DeliveryRefused> {
  const before = readFileSync(log);
  const error = (await rejected(call())) as DeliveryRefused;
  expect(error).toBeInstanceOf(DeliveryRefused);
  expect(error.reason).toBe(reason);
  expect(readFileSync(log).equals(before)).toBe(true);
  return error;
}

test("send refusals append nothing: unknown inbox, invalid value, closed inbox, ended run", async () => {
  const { log, flow } = await waitingChat();
  await refusedWith(log, () => send(log, "nope", "hi", { env, flow }), "unknown_inbox");
  const invalid = await refusedWith(log, () => send(log, "m", 3, { env, flow }), "invalid_value");
  expect(invalid.cause).toBeInstanceOf(TypeCheckError);
  expect(typeof (await close(log, "m", { env, flow }))).toBe("number");
  await refusedWith(log, () => send(log, "m", "hi", { env, flow }), "inbox_closed");
  expect((await advance({ log, flow, env })).status).toBe("succeeded");
  await refusedWith(log, () => send(log, "m", "hi", { env, flow }), "ended");
});

test("a send over the run's YIAN_MAX_EVENT_BYTES is too_large", async () => {
  const small = (name: string) => (name === "YIAN_MAX_EVENT_BYTES" ? "16" : env(name));
  const { log, flow } = await waitingChat(small);
  await refusedWith(log, () => send(log, "m", "a much longer event than sixteen bytes", { env: small, flow }), "too_large");
});

test("close twice: a seq, then null; a take waiting on the closed empty inbox gets closed: true", async () => {
  const { log, flow } = await waitingChat();
  expect(typeof (await close(log, "m", { env, flow }))).toBe("number");
  const before = readFileSync(log);
  expect(await close(log, "m", { env, flow })).toBeNull();
  expect(readFileSync(log).equals(before)).toBe(true);
  expect((await advance({ log, flow, env })).output).toEqual({ events: [], closed: true });
});

/** An ask handler that returns `pending()`. */
const pendingAsk = () => () => pending();

/** START -> a TOOL `ask` named `ask` with prompt "Ship?" (and `timeout` when given) -> END. */
function askFlow(timeout: string | null = null, extra: readonly Node[] = []): ComposedFlow {
  const ask = new ToolNode("ask", { toolId: "ask" });
  const wiring: Record<string, Record<string, unknown>> = { ask: { prompt: "Ship?", ...(timeout === null ? {} : { timeout }) } };
  ask.params = Object.keys(wiring.ask!).map((name) => new Parameter(name));
  for (const node of extra) wiring[node.id] = {};
  return new ComposedFlow(
    graph([ask, ...extra], [[START_ID, "ask"], ...extra.map((node) => [START_ID, node.id] as const)], {
      wiring,
      outputs: [new FlowOutput("answer", "${ask.output}")],
    }),
  );
}

/** A waiting run of `askFlow(timeout)` at a fresh log. */
async function waitingAsk(options: { timeout?: string | null; scheduler?: ReturnType<typeof handClock> } = {}) {
  const log = freshLog();
  const flow = askFlow(options.timeout ?? null);
  const tools = { ask: pendingAsk() };
  const result = await runFlow(flow, {}, { log, env, tools, scheduler: options.scheduler ?? null });
  expect(result.status).toBe("waiting");
  return { log, flow, tools, callId: `${result.runId}/ask` };
}

test("pendingCalls lists an open ask; complete answers it and it leaves the list", async () => {
  const { log, flow, tools, callId } = await waitingAsk();
  const pendingSeq = readRunLog(log).findIndex((envelope) => envelope.kind === "node_step");
  expect(pendingCalls(log)).toEqual([
    { callId, node: "ask", tool: "ask", arguments: { prompt: "Ship?" }, pendingSeq, deadline: null },
  ]);
  const seq = await complete(log, callId, { output: "yes" }, { env, flow });
  expect(readRunLog(log)[seq!]!.kind).toBe("node_step");
  expect(pendingCalls(log)).toEqual([]);
  expect((await advance({ log, flow, tools, env })).output).toBe("yes");
});

test("an identical repeat complete is a no-op; a different one is call_settled", async () => {
  const { log, flow, callId } = await waitingAsk();
  await complete(log, callId, { output: "yes" }, { env, flow });
  const before = readFileSync(log);
  expect(await complete(log, callId, { output: "yes" }, { env, flow })).toBeNull();
  expect(readFileSync(log).equals(before)).toBe(true);
  const error = await refusedWith(log, () => complete(log, callId, { output: "no" }, { env, flow }), "call_settled");
  expect(error.callState).toBe("answered");
});

test("complete refuses an unknown id and a take's id", async () => {
  const log = freshLog();
  const ask = new ToolNode("ask", { toolId: "ask" });
  ask.params = [new Parameter("prompt")];
  const flow = chatFlow([ask], { ask: { prompt: "Ship?" } });
  const result = await runFlow(flow, {}, { log, env, tools: { ask: pendingAsk() } });
  expect(result.status).toBe("waiting");
  await refusedWith(log, () => complete(log, "nope", { output: "x" }, { env, flow }), "unknown_call");
  await refusedWith(log, () => complete(log, `${result.runId}/heard`, { output: "x" }, { env, flow }), "not_host_call");
});

const T0 = Date.UTC(2026, 9, 24, 9, 0, 0);

test("a complete after an ask timed out and its node settled is ended", async () => {
  const { log, flow, tools, callId } = await waitingAsk({ timeout: "1s", scheduler: handClock(T0) });
  expect(pendingCalls(log)[0]!.deadline).toBe("2026-10-24T09:00:01.000Z");
  const advanced = await advance({ log, flow, tools, env, scheduler: handClock(T0 + 2000) });
  expect([advanced.status, advanced.output]).toEqual(["succeeded", null]);
  await refusedWith(log, () => complete(log, callId, { output: "late" }, { env, flow }), "ended");
});

test("cancel returns the open host calls and ends the run; later calls are ended", async () => {
  const { log, flow, tools, callId } = await waitingAsk();
  const open = pendingCalls(log);
  expect(await cancel(log, { env, flow })).toEqual(open);
  expect(readRunLog(log).at(-1)!.record).toBeInstanceOf(RunCancelledRecord);
  expect(pendingCalls(log)).toEqual([]);
  await refusedWith(log, () => complete(log, callId, { output: "yes" }, { env, flow }), "ended");
  await refusedWith(log, () => cancel(log, { env, flow }), "ended");
  expect((await advance({ log, flow, tools, env })).status).toBe("cancelled");
});

test("a node failure while an ask waits leaves pendingCalls empty", async () => {
  const log = freshLog();
  const boom = new ToolNode("boom", { toolId: "boom" });
  const flow = askFlow(null, [boom]);
  const failing = new FunctionTool({
    name: "boom",
    description: "fails",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => {
      throw new Error("boom");
    },
  });
  const result = await runFlow(flow, {}, { log, env, tools: { ask: pendingAsk(), boom: failing }, numWorkers: 0 });
  expect(result.status).toBe("failed");
  expect(readRunLog(log).some((envelope) => envelope.kind === "node_waiting")).toBe(true);
  expect(pendingCalls(log)).toEqual([]);
});

/** Write, in a fresh folder, a YAML flow whose CODE node runs `./fn.ts:up`, then a TOOL `ask`. */
async function codeAskFlow(): Promise<{ dir: string; flow: ComposedFlow }> {
  const dir = join(mkdtempSync(join(ROOT, "code-")), "flows");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "fn.ts"), "export function up(inputs: Record<string, any>): string {\n  return `up:${inputs.q}`;\n}\n");
  const text = [
    "id: root",
    "name: root",
    "nodes:",
    "  up:",
    "    kind: code",
    "    code: ./fn.ts:up",
    "    input:",
    "      q: hi",
    "    output: str",
    "  ask:",
    "    kind: tool",
    "    tool_id: ask",
    "    input:",
    "      prompt: ${up.output}",
    "output: ${ask.output}",
    "",
  ].join("\n");
  writeFileSync(join(dir, "flow.yaml"), text);
  return { dir, flow: await loadFlow(text, { searchPaths: [dir], sourceLabel: "flow.yaml" }) };
}

test("a refused delivery keeps the idle state: the next one needs no recompile", async () => {
  const { dir, flow } = await codeAskFlow();
  const log = freshLog();
  const result = await runFlow(flow, {}, { log, env, tools: { ask: pendingAsk() } });
  expect(result.status).toBe("waiting");
  unlinkSync(join(dir, "fn.ts"));
  await refusedWith(log, () => complete(log, "nope", { output: "x" }, { env }), "unknown_call");
  // Still a hit: nothing recompiles, so the missing CODE file isn't read.
  expect(typeof (await complete(log, `${result.runId}/ask`, { output: "yes" }, { env }))).toBe("number");
});

test("an offline delivery leaves an entry the next delivery reuses", async () => {
  const { dir, flow } = await codeAskFlow();
  const log = freshLog();
  const result = await runFlow(flow, {}, { log, env, tools: { ask: pendingAsk() } });
  await complete(log, `${result.runId}/ask`, { output: "yes" }, { env });
  unlinkSync(join(dir, "fn.ts"));
  // The entry the complete left is reused: the log is the file it left, so nothing recompiles.
  expect(await complete(log, `${result.runId}/ask`, { output: "yes" }, { env })).toBeNull();
});

describe("bad arguments and logs, for every host call", () => {
  const calls = {
    send: (log: string, options = {}) => send(log, "m", "hi", options),
    complete: (log: string, options = {}) => complete(log, "x", { output: "x" }, options),
    close: (log: string, options = {}) => close(log, "m", options),
    cancel: (log: string, options = {}) => cancel(log, options),
    pendingCalls: async (log: string) => pendingCalls(log),
  };

  for (const [name, call] of Object.entries(calls)) {
    test(`${name}: a missing log is ENOENT`, async () => {
      const error = (await rejected(call(join(ROOT, "missing.jsonl"), { env }))) as NodeJS.ErrnoException;
      expect(error.code).toBe("ENOENT");
    });

    test(`${name}: an empty log, or one not starting with run_started, is RangeError`, async () => {
      const empty = freshLog();
      writeFileSync(empty, "");
      expect(await rejected(call(empty, { env }))).toBeInstanceOf(RangeError);
      const { log } = await waitingChat();
      const headless = freshLog();
      writeFileSync(headless, readFileSync(log, "utf8").split("\n").slice(1).join("\n"));
      const before = readFileSync(headless);
      expect(await rejected(call(headless, { env }))).toBeInstanceOf(RangeError);
      expect(readFileSync(headless).equals(before)).toBe(true);
    });
  }

  for (const name of ["send", "complete", "close", "cancel"] as const) {
    test(`${name}: an offline env with another YIAN_* value is DefaultsMismatchError`, async () => {
      const { log, flow } = await waitingChat();
      const other = (key: string) => (key === "YIAN_MAX_EVENT_BYTES" ? "4096" : env(key));
      const before = readFileSync(log);
      expect(await rejected(calls[name](log, { env: other, flow }))).toBeInstanceOf(DefaultsMismatchError);
      expect(readFileSync(log).equals(before)).toBe(true);
    });
  }

  test("a non-string inbox or call id is TypeError", async () => {
    const { log, flow } = await waitingChat();
    expect(await rejected(send(log, 3 as never, "hi", { env, flow }))).toBeInstanceOf(TypeError);
    expect(await rejected(close(log, null as never, { env, flow }))).toBeInstanceOf(TypeError);
    expect(await rejected(complete(log, 3 as never, { output: "x" }, { env, flow }))).toBeInstanceOf(TypeError);
    expect(await rejected(complete(log, "x", "yes" as never, { env, flow }))).toBeInstanceOf(TypeError);
  });
});
