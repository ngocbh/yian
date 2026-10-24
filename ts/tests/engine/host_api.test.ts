/**
 * The host calls on a run log: `send`, `complete`, `close`, `cancel` and `pendingCalls`, live
 * through this process's engine or offline under the log's lock, with their typed refusals and
 * `RunBusyError`. TS-only: Python yian has no inboxes or pending calls.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { type ChildProcess, spawn } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { Readable } from "node:stream";
import { fileURLToPath } from "node:url";

import { FlowOutput, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FileLockError, lockFile } from "../../src/platform.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { DefaultsMismatchError } from "../../src/defaults.ts";
import { CallClosed, NodeSucceeded, NodeWaiting, type RunEvent, RunStarted, RunWaiting } from "../../src/events.ts";
import {
  advance,
  cancel,
  close,
  complete,
  DeliveryRefused,
  type DeliveryRefusal,
  type PendingCall,
  pendingCalls,
  RunBusyError,
  runFlow,
  send,
} from "../../src/run.ts";
import { PoolCommitRecord, readRunLog, RunCancelledRecord } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { inboxType, Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { gateTool, handClock } from "./inbox_fixtures.ts";
import { processChatFlow, processEnv } from "./yian-host-api-child-hold.ts";
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
  // The ask declares no `output:`, which reads as `str`, so its timeout fails it.
  expect([advanced.status, advanced.failure?.code]).toEqual(["failed", "tool.timeout"]);
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

test("runFlow: a send from onEvent(RunStarted) reaches the engine", async () => {
  const log = freshLog();
  let sent: Promise<number> | null = null;
  const result = await runFlow(chatFlow(), {}, {
    log,
    env,
    onEvent: (event: RunEvent) => {
      if (event instanceof RunStarted) sent = send(log, "m", "hi");
    },
  });
  expect([result.status, result.output]).toEqual(["succeeded", { events: ["hi"], closed: false }]);
  const seq = await sent!;
  expect(readRunLog(log)[seq]!.record).toBeInstanceOf(PoolCommitRecord);
});

test("a send during a live run wakes the take while another node runs", async () => {
  const log = freshLog();
  const gate = gateTool();
  let sent: Promise<number> | null = null;
  const result = await runFlow(chatFlow([new ToolNode("gate", { toolId: "gate" })]), {}, {
    log,
    env,
    numWorkers: 2,
    tools: { gate: gate.tool },
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeWaiting && event.nodeId === "heard") sent = send(log, "m", "hi");
      // The gate is held until the take has woken and committed.
      if (event instanceof NodeSucceeded && event.nodeId === "heard") gate.release();
    },
  });
  expect([result.status, result.output]).toEqual(["succeeded", { events: ["hi"], closed: false }]);
  const kinds = readRunLog(log).map((envelope) => envelope.kind);
  expect(kinds).toContain("node_woken");
  expect(kinds).not.toContain("run_waiting");
  expect(typeof (await sent!)).toBe("number");
});

test("cancel during a live run returns the open ask and cancels the run", async () => {
  const log = freshLog();
  const askWaiting = Promise.withResolvers<void>();
  let open = null as readonly PendingCall[] | null;
  const canceller = new FunctionTool({
    name: "gate",
    description: "cancels the run once the ask waits",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: async () => {
      await askWaiting.promise;
      open = await cancel(log);
      return "done";
    },
  });
  const result = await runFlow(askFlow(null, [new ToolNode("gate", { toolId: "gate" })]), {}, {
    log,
    env,
    numWorkers: 8,
    tools: { ask: pendingAsk(), gate: canceller },
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeWaiting && event.nodeId === "ask") askWaiting.resolve();
    },
  });
  expect(result.status).toBe("cancelled");
  expect(open).toEqual([expect.objectContaining({ callId: `${result.runId}/ask`, node: "ask", tool: "ask" })]);
  expect(result.events.filter((event) => event instanceof CallClosed)).toEqual([
    new CallClosed({ callId: `${result.runId}/ask`, reason: "cancelled" }),
  ]);
});

test("pendingCalls reads a log another writer holds", async () => {
  const { log, callId } = await waitingAsk();
  const holder = lockFile(`${realpathSync(log)}.lock`);
  try {
    expect(pendingCalls(log).map((call) => call.callId)).toEqual([callId]);
  } finally {
    holder.release();
  }
});

test("codeRoot: an offline call refuses a CODE file outside it before any append; inside, it appends", async () => {
  const { dir, flow } = await codeAskFlow();
  const root = join(dir, "..");
  const out = mkdtempSync(join(ROOT, "out-"));
  const log = freshLog();
  const result = await runFlow(flow, {}, { log, env, tools: { ask: pendingAsk() } });
  const code = readFileSync(join(dir, "fn.ts"));
  renameSync(join(dir, "fn.ts"), join(out, "fn.ts"));
  symlinkSync(join(out, "fn.ts"), join(dir, "fn.ts"));
  const callId = `${result.runId}/ask`;
  const options = { env, codeRoot: root };
  const before = readFileSync(log);
  for (const call of [
    () => send(log, "m", "hi", options),
    () => complete(log, callId, { output: "yes" }, options),
    () => close(log, "m", options),
    () => cancel(log, options),
  ]) {
    const error = await rejected(call());
    expect(error).toBeInstanceOf(LoadError);
    expect((error as Error).message).toEndWith(`outside the code root '${realpathSync(root)}'`);
    expect(readFileSync(log).equals(before)).toBe(true);
  }
  unlinkSync(join(dir, "fn.ts"));
  writeFileSync(join(dir, "fn.ts"), code);
  expect(typeof (await complete(log, callId, { output: "yes" }, options))).toBe("number");
  expect((await cancel(log, options)).map((call) => call.callId)).toEqual([]);
  expect(readRunLog(log).at(-1)!.record).toBeInstanceOf(RunCancelledRecord);
});

const HOLD_SCRIPT = fileURLToPath(new URL("./yian-host-api-child-hold.ts", import.meta.url));
const SEND_SCRIPT = fileURLToPath(new URL("./yian-host-api-child-send.ts", import.meta.url));
// Without `--no-install`, Bun may try to install the script's imports from the network.
const FLAGS = process.versions.bun === undefined ? [] : ["--no-install"];

/** Spawn the child `script` with `args`, piping its stdio, killed after 20 seconds. */
function child(script: string, args: readonly string[]): ChildProcess {
  return spawn(process.execPath, [...FLAGS, script, ...args], { stdio: ["pipe", "pipe", "pipe"], timeout: 20_000 });
}

/** The first line `stream` writes; rejects if it closes or errors first. */
function firstLine(stream: Readable): Promise<string> {
  return new Promise((resolve, reject) => {
    let text = "";
    const onData = (chunk: Buffer) => {
      text += chunk.toString("utf8");
      const end = text.indexOf("\n");
      if (end >= 0) {
        stream.off("data", onData);
        resolve(text.slice(0, end));
      }
    };
    stream.on("data", onData);
    stream.once("close", () => reject(new Error(`the stream closed before a line: ${JSON.stringify(text)}`)));
    stream.once("error", reject);
  });
}

/** The child's exit code, once it closes (resolves on `close`; rejects on `error`). */
function exited(proc: ChildProcess): Promise<number | null> {
  return new Promise((resolve, reject) => {
    if (proc.exitCode !== null || proc.signalCode !== null) {
      resolve(proc.exitCode);
      return;
    }
    proc.once("close", (code) => resolve(code));
    proc.once("error", reject);
  });
}

/** Kill `proc` unless it has exited. */
function reap(proc: ChildProcess): void {
  if (proc.exitCode === null && proc.signalCode === null) proc.kill("SIGKILL");
}

test("a send while another process moves the run is RunBusyError; after it ends, the send lands", async () => {
  const log = freshLog();
  const proc = child(HOLD_SCRIPT, [log]);
  try {
    expect(await firstLine(proc.stdout!)).toBe("locked");
    const flow = processChatFlow();
    expect(await rejected(send(log, "m", "hi", { env: processEnv, flow }))).toBeInstanceOf(RunBusyError);
    proc.stdin!.end("go\n");
    // Its run ended waiting on the take.
    expect(await exited(proc)).toBe(0);
    expect(typeof (await send(log, "m", "hi", { env: processEnv, flow }))).toBe("number");
    expect((await advance({ log, flow, env: processEnv, tools: { gate: openGate() } })).output).toEqual({
      events: ["hi"],
      closed: false,
    });
  } finally {
    reap(proc);
  }
}, 30_000);

/** A `gate` tool that opens at once. */
function openGate(): FunctionTool {
  return new FunctionTool({
    name: "gate",
    description: "opens at once",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "opened",
  });
}

test("a send from another process to a run this one left waiting: advance rebuilds and reads it", async () => {
  const log = freshLog();
  const flow = processChatFlow();
  const tools = { gate: openGate() };
  expect((await runFlow(flow, {}, { log, env: processEnv, tools })).status).toBe("waiting");
  const proc = child(SEND_SCRIPT, [log, "send"]);
  try {
    proc.stdin!.end();
    const line = firstLine(proc.stdout!);
    expect(await exited(proc)).toBe(0);
    expect(typeof JSON.parse(await line)).toBe("number");
    const advanced = await advance({ log, flow, env: processEnv, tools });
    expect([advanced.status, advanced.output]).toEqual(["succeeded", { events: ["from child"], closed: false }]);
  } finally {
    reap(proc);
  }
}, 30_000);

test("registry hit: after advance waits, a call needs no recompile until another process appends", async () => {
  const { dir, flow } = await codeAskFlow();
  const log = freshLog();
  const tools = { ask: pendingAsk() };
  const result = await runFlow(flow, {}, { log, env, tools });
  expect((await advance({ log, env, tools })).status).toBe("waiting");
  const code = readFileSync(join(dir, "fn.ts"));
  unlinkSync(join(dir, "fn.ts"));
  // A hit: nothing recompiles, so the missing CODE file isn't read.
  expect(typeof (await complete(log, `${result.runId}/ask`, { output: "yes" }, { env }))).toBe("number");
  writeFileSync(join(dir, "fn.ts"), code);
  const proc = child(SEND_SCRIPT, [log, "cancel"]);
  try {
    proc.stdin!.end();
    const line = firstLine(proc.stdout!);
    expect(await exited(proc)).toBe(0);
    expect(JSON.parse(await line)).toEqual([]);
  } finally {
    reap(proc);
  }
  unlinkSync(join(dir, "fn.ts"));
  // The other process changed the log: the miss recompiles, and the CODE file is gone.
  expect(await rejected(complete(log, `${result.runId}/ask`, { output: "yes" }, { env }))).toBeInstanceOf(LoadError);
}, 30_000);

/** Write, in a fresh folder, a YAML flow whose CODE node runs `./fn.ts:up` beside a take from inbox `m`. */
async function codeTakeFlow(): Promise<{ dir: string; flow: ComposedFlow }> {
  const dir = join(mkdtempSync(join(ROOT, "code-take-")), "flows");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "fn.ts"), "export function up(): string {\n  return \"up\";\n}\n");
  const text = [
    "id: root",
    "name: root",
    "input:",
    "  m: !Inbox(str)",
    "nodes:",
    "  up:",
    "    kind: code",
    "    code: ./fn.ts:up",
    "    output: str",
    "  heard:",
    "    kind: tool",
    "    tool_id: take",
    "    input:",
    "      from: ${input.m}",
    "      take: all",
    "output:",
    "  up: ${up.output}",
    "  heard: ${heard.output}",
    "",
  ].join("\n");
  writeFileSync(join(dir, "flow.yaml"), text);
  return { dir, flow: await loadFlow(text, { searchPaths: [dir], sourceLabel: "flow.yaml" }) };
}

test("YAML: a refused send keeps the idle state, so a good send needs no recompile", async () => {
  const { dir, flow } = await codeTakeFlow();
  const log = freshLog();
  expect((await runFlow(flow, {}, { log, env })).status).toBe("waiting");
  unlinkSync(join(dir, "fn.ts"));
  await refusedWith(log, () => send(log, "nope", "hi", { env }), "unknown_inbox");
  // Still a hit: nothing recompiles, so the missing CODE file isn't read.
  expect(typeof (await send(log, "m", "hi", { env }))).toBe("number");
  expect(typeof (await close(log, "m", { env }))).toBe("number");
});

test("YAML codeRoot: send and close append once the CODE file is inside the root", async () => {
  const { dir, flow } = await codeTakeFlow();
  const root = join(dir, "..");
  const out = mkdtempSync(join(ROOT, "out-"));
  const log = freshLog();
  expect((await runFlow(flow, {}, { log, env })).status).toBe("waiting");
  const code = readFileSync(join(dir, "fn.ts"));
  renameSync(join(dir, "fn.ts"), join(out, "fn.ts"));
  symlinkSync(join(out, "fn.ts"), join(dir, "fn.ts"));
  const before = readFileSync(log);
  for (const call of [() => send(log, "m", "hi", { env, codeRoot: root }), () => close(log, "m", { env, codeRoot: root })]) {
    expect(await rejected(call())).toBeInstanceOf(LoadError);
    expect(readFileSync(log).equals(before)).toBe(true);
  }
  unlinkSync(join(dir, "fn.ts"));
  writeFileSync(join(dir, "fn.ts"), code);
  expect(typeof (await send(log, "m", "hi", { env, codeRoot: root }))).toBe("number");
  expect(typeof (await close(log, "m", { env, codeRoot: root }))).toBe("number");
  expect((await advance({ log, env, codeRoot: root })).output).toEqual({ up: "up", heard: { events: ["hi"], closed: true } });
});

for (const numWorkers of [0, 2]) {
  test(`numWorkers ${numWorkers}: a send as the run starts waiting lands, and is read live or by the next advance`, async () => {
    const log = freshLog();
    const flow = chatFlow();
    let sent: Promise<number> | null = null;
    const result = await runFlow(flow, {}, {
      log,
      env,
      numWorkers,
      onEvent: (event: RunEvent) => {
        if (event instanceof NodeWaiting && event.nodeId === "heard") sent = send(log, "m", "hi", { env, flow });
      },
    });
    const seq = await sent!;
    expect(readRunLog(log)[seq]!.record).toBeInstanceOf(PoolCommitRecord);
    const done = result.status === "waiting" ? await advance({ log, flow, env }) : result;
    expect([done.status, done.output]).toEqual(["succeeded", { events: ["hi"], closed: false }]);
  });

  test(`numWorkers ${numWorkers}: a send and an advance just after the loop ended wait for the run's close, then the advance reads the send`, async () => {
    const log = freshLog();
    const flow = chatFlow();
    let sent: Promise<number> | null = null;
    let advanced: Promise<Awaited<ReturnType<typeof advance>>> | null = null;
    const result = await runFlow(flow, {}, {
      log,
      env,
      numWorkers,
      onEvent: (event: RunEvent) => {
        if (event instanceof RunWaiting) {
          // A microtask later the loop has ended, but runFlow hasn't closed the run yet.
          const gap = Promise.resolve();
          sent = gap.then(() => send(log, "m", "hi", { env, flow }));
          advanced = gap.then(() => advance({ log, flow, env }));
        }
      },
    });
    expect(result.status).toBe("waiting");
    await sent!;
    const done = await advanced!;
    expect([done.status, done.output]).toEqual(["succeeded", { events: ["hi"], closed: false }]);
  });
}

test("complete refuses {output: undefined}", async () => {
  const { log, flow, callId } = await waitingAsk();
  expect(await rejected(complete(log, callId, { output: undefined }, { env, flow }))).toBeInstanceOf(TypeError);
});
