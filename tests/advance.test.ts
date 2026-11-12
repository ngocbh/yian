/**
 * `advance` and the run boundary's waiting runs: the waiting result, when `advance` appends, the
 * lock released while a run waits, and the in-process registry of waiting runs.
 */
import { afterAll, expect, spyOn, test } from "bun:test";
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

import { ComposedFlow } from "../src/compose/build/pipeline.ts";
import { LoadError } from "../src/compose/helpers/errors.ts";
import { loadFlow } from "../src/compose/loader.ts";
import { NodeStarted, type RunEvent, RunStarted } from "../src/events.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { Parameter } from "../src/nodes/contracts.ts";
import { ToolNode } from "../src/nodes/tool/node.ts";
import { lockFile, withWarnings } from "../src/platform.ts";
import { advance, close, replayRunLog, runFlow, send } from "../src/run.ts";
import {
  isRunLogLocked,
  JsonlRunLog,
  NodeWokenRecord,
  readRunLog,
  RunResumedRecord,
  RunStoppedRecord,
  RunWaitingRecord,
} from "../src/runtime/log.ts";
import { commitPrepared, prepareRunCancelled } from "../src/runtime/state.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import { completeByHand, gateTool, handClock } from "./helpers/inbox_fixtures.ts";
import { crashAfter, graph, NO_ENV, rejected, serving, textReply } from "./helpers/run_state_fixtures.ts";
import { YianTypeError } from "../src/typesys/errors.ts";

const ROOT = mkdtempSync(join(tmpdir(), "advance-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

/** A host tool `lookup` whose handler returns `pending()`. */
function pendingTool(): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "looks something up later",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => pending(),
  });
}

/** START -> a TOOL node `ask` calling `lookup` -> END. */
function hostFlow(): ComposedFlow {
  return new ComposedFlow(graph([new ToolNode("ask", { toolId: "lookup" })], [["__start__", "ask"], ["ask", "__end__"]]));
}

test("runFlow ends waiting: status, waits, no output, the lock released", async () => {
  const log = freshLog();
  const result = await runFlow(hostFlow(), {}, { log, env: NO_ENV, tools: { lookup: pendingTool() }, numWorkers: 0 });
  expect(result.status).toBe("waiting");
  expect(result.waits).toEqual([{ node: "ask", callId: `${result.runId}/ask`, tool: "lookup", inbox: null, deadline: null, settled: false }]);
  expect([result.output, result.unread.size, isRunLogLocked(log)]).toEqual([null, 0, false]);
});

const T0 = Date.UTC(2026, 9, 24, 9, 0, 0);

/** The record kinds of the log at `log`, in order. */
const kinds = (log: string) => readRunLog(log).map((envelope) => envelope.kind);

/** A waiting run of `hostFlow()` at a fresh log, with what advancing it needs. */
async function waitingRun(options: { numWorkers?: number } = {}) {
  const log = freshLog();
  const flow = hostFlow();
  const tools = { lookup: pendingTool() };
  const result = await runFlow(flow, {}, { log, env: NO_ENV, tools, numWorkers: options.numWorkers ?? 0 });
  expect(result.status).toBe("waiting");
  return { log, flow, tools, callId: `${result.runId}/ask` };
}

test("advance on a waiting run that is neither ready nor due appends nothing", async () => {
  const { log, flow, tools } = await waitingRun();
  const before = readFileSync(log);
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect([result.status, result.events]).toEqual(["waiting", []]);
  expect(readFileSync(log).equals(before)).toBe(true);
  expect(isRunLogLocked(log)).toBe(false);
});

test("advance on a ready waiting run wakes the node, with no run_resumed", async () => {
  const { log, flow, tools, callId } = await waitingRun();
  completeByHand(log, flow, callId, { output: "yes" });
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("succeeded");
  const answer = readRunLog(log).find((envelope) => envelope.kind === "pool_commit" && envelope.payload["node_id"] === "ask");
  expect(answer?.payload["value"]).toBe("yes");
  expect(kinds(log)).not.toContain("run_resumed");
  expect(kinds(log)).toContain("node_woken");
});

test("advance wakes a due sleep with timed_out", async () => {
  const log = freshLog();
  const nap = new ToolNode("nap", { toolId: "sleep" });
  nap.params = [new Parameter("for")];
  const flow = new ComposedFlow(graph([nap], [["__start__", "nap"], ["nap", "__end__"]], { wiring: { nap: { for: 60n } } }));
  const first = await runFlow(flow, {}, { log, env: NO_ENV, numWorkers: 0, scheduler: handClock(T0) });
  expect(first.status).toBe("waiting");
  const result = await advance({ log, flow, env: NO_ENV, scheduler: handClock(T0 + 61_000) });
  expect(result.status).toBe("succeeded");
  const woken = readRunLog(log).find((envelope) => envelope.record instanceof NodeWokenRecord)!.record as NodeWokenRecord;
  expect(woken.timedOut).toEqual([`${first.runId}/nap`]);
  expect(kinds(log)).not.toContain("run_resumed");
});

test("advance after a crash between node_waiting and run_waiting resumes, then waits", async () => {
  const probe = await waitingRun();
  const k = kinds(probe.log).indexOf("run_waiting");
  const log = freshLog();
  const flow = hostFlow();
  const tools = { lookup: pendingTool() };
  await crashAfter(k, () => runFlow(flow, {}, { log, env: NO_ENV, tools, numWorkers: 0 }));
  expect(kinds(log).at(-1)).toBe("node_waiting");
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("waiting");
  expect(kinds(log).slice(-2)).toEqual(["run_resumed", "run_waiting"]);
  const resumed = readRunLog(log).at(-2)!.record as RunResumedRecord;
  expect(resumed.retryNodeIds).toEqual([]);
});

test("advance with another worker count writes run_resumed", async () => {
  const { log, flow, tools } = await waitingRun();
  const result = await advance({ log, flow, tools, env: NO_ENV, numWorkers: 2 });
  expect(result.status).toBe("waiting");
  expect(kinds(log).slice(-2)).toEqual(["run_resumed", "run_waiting"]);
});

test.each([
  [{ onEvent: 5 }, "onEvent must be a function or null"],
  [{ onStream: 5 }, "onStream must be a function or null"],
])("a callback that isn't a function is refused before the log changes (%#)", async (option, message) => {
  const { log, flow, tools } = await waitingRun();
  const before = readFileSync(log);
  // Another worker count would write run_resumed.
  const options = { log, flow, tools, env: NO_ENV, numWorkers: 2, ...option } as never;
  expect(await rejected(advance(options))).toEqual(new YianTypeError(message));
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("advance after a stop while a node waits retries only the running node", async () => {
  const log = freshLog();
  const flow = new ComposedFlow(
    graph(
      [new ToolNode("ask", { toolId: "lookup" }), new ToolNode("gate", { toolId: "gate" })],
      [["__start__", "ask"], ["__start__", "gate"], ["ask", "__end__"], ["gate", "__end__"]],
    ),
  );
  const gate = gateTool();
  const tools = { lookup: pendingTool(), gate: gate.tool };
  const host = new AbortController();
  const stop = new Error("host stop");
  void gate.entered.then(() => host.abort(stop));
  expect(await rejected(runFlow(flow, {}, { log, env: NO_ENV, tools, numWorkers: 0, signal: host.signal }))).toBe(stop);
  const stopped = readRunLog(log).find((envelope) => envelope.record instanceof RunStoppedRecord)!.record as RunStoppedRecord;
  expect(stopped.nodeIds).toEqual(["gate"]);
  gate.release();
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("waiting");
  const resumed = readRunLog(log).find((envelope) => envelope.record instanceof RunResumedRecord)!.record as RunResumedRecord;
  expect(resumed.retryNodeIds).toEqual(["gate"]);
});

test("advance on a cancelled log returns cancelled", async () => {
  const { log, flow, tools } = await waitingRun();
  const authority = JsonlRunLog.resume(log);
  try {
    const state = replayRunLog(flow, authority);
    commitPrepared(authority, state, prepareRunCancelled(state));
  } finally {
    authority.close();
  }
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("cancelled");
});

test("advance on a missing log rejects with ENOENT", async () => {
  const error = await rejected(advance({ log: join(ROOT, "missing.jsonl"), env: NO_ENV }));
  expect((error as NodeJS.ErrnoException).code).toBe("ENOENT");
});

const BUSY = (log: string) => `run log ${log} is locked by another writer (is the run still active?)`;

/** START fans out to host call `ask` and gate `gate`, both feeding END. */
function gateFlow(): ComposedFlow {
  return new ComposedFlow(
    graph(
      [new ToolNode("ask", { toolId: "lookup" }), new ToolNode("gate", { toolId: "gate" })],
      [["__start__", "ask"], ["__start__", "gate"], ["ask", "__end__"], ["gate", "__end__"]],
    ),
  );
}

test("an in-process advance while runFlow moves the run is busy", async () => {
  const log = freshLog();
  const flow = gateFlow();
  const gate = gateTool();
  const tools = { lookup: pendingTool(), gate: gate.tool };
  let attempt: Promise<unknown> = Promise.resolve(null);
  const running = runFlow(flow, {}, {
    log,
    env: NO_ENV,
    tools,
    numWorkers: 0,
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeStarted && event.nodeId === "gate") attempt = rejected(advance({ log, flow, tools, env: NO_ENV }));
    },
  });
  await gate.entered;
  expect(((await attempt) as Error).message).toBe(BUSY(resolve(log)));
  gate.release();
  expect((await running).status).toBe("waiting");
});

test("a second advance waits for the first and then reads its result", async () => {
  const { log, flow, tools, callId } = await waitingRun();
  completeByHand(log, flow, callId, { output: "yes" });
  const aEntered = Promise.withResolvers<void>();
  const aGate = Promise.withResolvers<void>();
  const a = advance({
    log,
    flow,
    tools,
    env: NO_ENV,
    checkLlmConfigs: async () => {
      aEntered.resolve();
      await aGate.promise;
    },
  });
  await aEntered.promise;
  const b = advance({ log, flow, tools, env: NO_ENV });
  aGate.resolve();
  expect((await a).status).toBe("succeeded");
  const length = readRunLog(log).length;
  const second = await b;
  expect([second.status, second.events]).toEqual(["succeeded", []]);
  expect(readRunLog(log)).toHaveLength(length);
});

test("an advance from onEvent(RunStarted) waits for runFlow, not on the lock", async () => {
  const log = freshLog();
  const flow = hostFlow();
  const tools = { lookup: pendingTool() };
  let attempt: Promise<Awaited<ReturnType<typeof advance>>> | null = null;
  const result = await runFlow(flow, {}, {
    log,
    env: NO_ENV,
    tools,
    numWorkers: 0,
    onEvent: (event: RunEvent) => {
      if (event instanceof RunStarted) attempt = advance({ log, flow, tools, env: NO_ENV });
    },
  });
  expect(result.status).toBe("waiting");
  const advanced = await attempt!;
  expect([advanced.status, advanced.events]).toEqual(["waiting", []]);
});

test("a process-local run refuses a host call's pending(): nothing could answer it", async () => {
  const result = await runFlow(hostFlow(), {}, { log: null, env: NO_ENV, tools: { lookup: pendingTool() } });
  expect([result.status, result.log]).toEqual(["failed", null]);
  expect(result.failure?.message).toContain("returned pending() outside a durable run; give the run a log");
});

test("a bad option holds nothing", async () => {
  const { log, flow, tools } = await waitingRun();
  expect(await rejected(advance({ log, flow: 3 as never, env: NO_ENV }))).toBeInstanceOf(TypeError);
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("waiting");
});

test("a busy refusal releases the hold", async () => {
  const { log, flow, tools } = await waitingRun();
  const holder = lockFile(`${realpathSync(log)}.lock`);
  try {
    expect(((await rejected(advance({ log, flow, tools, env: NO_ENV }))) as Error).message).toBe(BUSY(resolve(log)));
  } finally {
    holder.release();
  }
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("waiting");
});

test("a log-close failure after the call is a warning, and releases the hold and the lock", async () => {
  const { log, flow, tools } = await waitingRun();
  const close = JsonlRunLog.prototype.close;
  const spy = spyOn(JsonlRunLog.prototype, "close").mockImplementationOnce(function (this: JsonlRunLog) {
    close.call(this);
    throw new Error("close failed");
  });
  const warnings: string[] = [];
  try {
    const result = await withWarnings(
      (message) => warnings.push(message),
      () => advance({ log, flow, tools, env: NO_ENV }),
    );
    expect(result.status).toBe("waiting");
  } finally {
    spy.mockRestore();
  }
  expect(warnings).toEqual([`run log ${resolve(log)} holds this call's records, but closing it failed: close failed`]);
  expect(isRunLogLocked(log)).toBe(false);
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("waiting");
});

test("a log-close failure after runFlow wrote its run is a warning; the result stands", async () => {
  const log = freshLog();
  const close = JsonlRunLog.prototype.close;
  const spy = spyOn(JsonlRunLog.prototype, "close").mockImplementationOnce(function (this: JsonlRunLog) {
    close.call(this);
    throw new Error("close failed");
  });
  const warnings: string[] = [];
  try {
    const result = await withWarnings(
      (message) => warnings.push(message),
      () => runFlow(hostFlow(), {}, { log, env: NO_ENV, tools: { lookup: pendingTool() }, numWorkers: 0 }),
    );
    expect(result.status).toBe("waiting");
  } finally {
    spy.mockRestore();
  }
  expect(warnings).toEqual([`run log ${resolve(log)} holds this call's records, but closing it failed: close failed`]);
  expect(isRunLogLocked(log)).toBe(false);
  expect(readRunLog(log).at(-1)!.record).toBeInstanceOf(RunWaitingRecord);
});

/** Write, in a fresh folder, a flow whose CODE node runs `./fn.ts:up`, then a TOOL `ask` calling `lookup`. */
async function codeHostFlow(root: string, codeRoot: string | null = null): Promise<{ dir: string; flow: ComposedFlow }> {
  const dir = join(root, "flows");
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
    "    tool_id: lookup",
    "    input:",
    "      q: ${up.output}",
    "output: ${ask.output}",
    "",
  ].join("\n");
  writeFileSync(join(dir, "flow.yaml"), text);
  return { dir, flow: await loadFlow(text, { searchPaths: [dir], sourceLabel: "flow.yaml", codeRoot }) };
}

/** `lookup` taking `q`, whose handler returns `pending()`. */
function lookupQ(): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "looks something up later",
    parametersJsonSchema: { type: "object", properties: { q: { type: "string" } } },
    execute: () => pending(),
  });
}

test("a waiting run's registry entry saves the recompile until the log changes", async () => {
  const root = mkdtempSync(join(ROOT, "hit-"));
  const { dir, flow } = await codeHostFlow(root);
  const log = freshLog();
  const tools = { lookup: lookupQ() };
  const first = await runFlow(flow, {}, { log, env: NO_ENV, tools });
  expect(first.status).toBe("waiting");
  unlinkSync(join(dir, "fn.ts"));
  // A hit: nothing recompiles, so the missing CODE file isn't read.
  expect((await advance({ log, tools, env: NO_ENV })).status).toBe("waiting");
  completeByHand(log, flow, `${first.runId}/ask`, { output: "yes" });
  // The outside append changed the log: the miss recompiles, and the CODE file is gone.
  expect(await rejected(advance({ log, tools, env: NO_ENV }))).toBeInstanceOf(LoadError);
});

test("runFlow's entry of a run started under codeRoot serves an advance passing that root, or none", async () => {
  const root = mkdtempSync(join(ROOT, "root-"));
  const { dir, flow } = await codeHostFlow(root, root);
  const log = freshLog();
  const tools = { lookup: lookupQ() };
  expect((await runFlow(flow, {}, { log, env: NO_ENV, tools })).status).toBe("waiting");
  unlinkSync(join(dir, "fn.ts"));
  // Hits: nothing recompiles, so the missing CODE file isn't read.
  expect((await advance({ log, tools, env: NO_ENV, codeRoot: root })).status).toBe("waiting");
  expect((await advance({ log, tools, env: NO_ENV })).status).toBe("waiting");
});

test("advance of a run started under codeRoot refuses a CODE file outside it with no codeRoot passed", async () => {
  const root = mkdtempSync(join(ROOT, "root-"));
  const out = mkdtempSync(join(ROOT, "out-"));
  const { dir, flow } = await codeHostFlow(root, root);
  const log = freshLog();
  const tools = { lookup: lookupQ() };
  const first = await runFlow(flow, {}, { log, env: NO_ENV, tools });
  expect(first.status).toBe("waiting");
  const code = readFileSync(join(dir, "fn.ts"));
  renameSync(join(dir, "fn.ts"), join(out, "fn.ts"));
  symlinkSync(join(out, "fn.ts"), join(dir, "fn.ts"));
  // The outside append changed the log, so the next advance recompiles.
  completeByHand(log, flow, `${first.runId}/ask`, { output: "yes" });
  const before = readFileSync(log);
  const error = await rejected(advance({ log, tools, env: NO_ENV }));
  expect(error).toBeInstanceOf(LoadError);
  expect((error as Error).message).toEndWith(" is outside the code root");
  expect(readFileSync(log).equals(before)).toBe(true);
  unlinkSync(join(dir, "fn.ts"));
  writeFileSync(join(dir, "fn.ts"), code);
  expect((await advance({ log, tools, env: NO_ENV })).status).toBe("succeeded");
});

test("an advance passing an API key the flow doesn't use still hits the registry with that key", async () => {
  const root = mkdtempSync(join(ROOT, "key-"));
  const { dir, flow } = await codeHostFlow(root);
  const log = freshLog();
  const tools = { lookup: lookupQ() };
  expect((await runFlow(flow, {}, { log, env: NO_ENV, tools })).status).toBe("waiting");
  // runFlow's entry holds no key, so this one recompiles and leaves an entry holding the key.
  expect((await advance({ log, tools, env: NO_ENV, apiKey: "sk-test" })).status).toBe("waiting");
  unlinkSync(join(dir, "fn.ts"));
  // A hit: nothing recompiles, so the missing CODE file isn't read.
  expect((await advance({ log, tools, env: NO_ENV, apiKey: "sk-test" })).status).toBe("waiting");
});

test("a runFlow given an API key the flow doesn't use leaves an entry an advance with that key hits", async () => {
  const root = mkdtempSync(join(ROOT, "run-key-"));
  const { dir, flow } = await codeHostFlow(root);
  const log = freshLog();
  const tools = { lookup: lookupQ() };
  const llmConfig = { provider: "openai", api_key: "sk-test" };
  expect((await runFlow(flow, {}, { log, env: NO_ENV, tools, llmConfig })).status).toBe("waiting");
  unlinkSync(join(dir, "fn.ts"));
  // A hit: nothing recompiles, so the missing CODE file isn't read.
  expect((await advance({ log, tools, env: NO_ENV, apiKey: "sk-test" })).status).toBe("waiting");
});

test("repeated advances of a waiting run that can't move each append nothing", async () => {
  const { log, flow, tools } = await waitingRun();
  const before = readFileSync(log);
  for (let i = 0; i < 3; i++) {
    const result = await advance({ log, flow, tools, env: NO_ENV });
    expect([result.status, result.events]).toEqual(["waiting", []]);
    expect(readFileSync(log).equals(before)).toBe(true);
  }
});

test("advancing a waiting host call never calls its handler again", async () => {
  const log = freshLog();
  const flow = hostFlow();
  let calls = 0;
  const lookup = new FunctionTool({
    name: "lookup",
    description: "looks something up later",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => {
      calls += 1;
      return pending();
    },
  });
  const tools = { lookup };
  const first = await runFlow(flow, {}, { log, env: NO_ENV, tools });
  expect([first.status, calls]).toEqual(["waiting", 1]);
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("waiting");
  // A rebuilt state (another worker count resumes the run) doesn't call it either.
  expect((await advance({ log, flow, tools, env: NO_ENV, numWorkers: 2 })).status).toBe("waiting");
  completeByHand(log, flow, `${first.runId}/ask`, { output: "yes" });
  const done = await advance({ log, flow, tools, env: NO_ENV });
  expect([done.status, calls]).toEqual(["succeeded", 1]);
});

/** A flow whose agent summarises the notes its take reads from the `Inbox(str)` input `notes`. */
const NOTES_AGENT = `id: notes
name: notes
input:
  notes: !Inbox(str)
nodes:
  got:
    kind: tool
    tool_id: take
    input:
      from: \${input.notes}
      take: all
  hello:
    kind: agent
    input:
      notes: \${got.output.events}
    prompt: Summarise \${notes}.
    output: str
output: \${hello.output}
`;

/** A run of `NOTES_AGENT` started with a host key (and `headers`), whose inbox got a note and closed. */
async function keyedNotes(headers: Record<string, string> | null = null) {
  const flow = await loadFlow(NOTES_AGENT, { sourceLabel: "notes.yaml" });
  const log = freshLog();
  const llmConfig = { provider: "openai", model: "gpt-test", api_key: "sk-test", ...(headers === null ? {} : { headers }) };
  const first = await runFlow(flow, {}, { log, env: NO_ENV, llmConfig, models: serving(new ScriptedModel([])) });
  expect(first.status).toBe("waiting");
  await send(log, "notes", "buy milk", { env: NO_ENV });
  await close(log, "notes", { env: NO_ENV });
  return log;
}

test("advance without the run's host key is refused before anything is appended; with it the agent runs", async () => {
  const log = await keyedNotes();
  const before = readFileSync(log);
  const model = new ScriptedModel([textReply("milk")]);
  const error = await rejected(advance({ log, env: NO_ENV, models: serving(model) }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(
    "this run's openai API key came from the host's llmConfig.api_key and is not recorded; pass it " +
      "again: advance({ log, apiKey })",
  );
  expect(readFileSync(log).equals(before)).toBe(true);
  expect(model.requests).toEqual([]);
  const resumed = await advance({ log, env: NO_ENV, apiKey: "sk-test", models: serving(model) });
  expect([resumed.status, resumed.output]).toEqual(["succeeded", "milk"]);
});

test("advance without the host key of a run whose key came with headers names both", async () => {
  const log = await keyedNotes({ "X-Team": "a" });
  const before = readFileSync(log);
  const error = await rejected(advance({ log, env: NO_ENV, headers: { "X-Team": "a" }, models: serving(new ScriptedModel([])) }));
  expect((error as Error).message).toEndWith("pass it again: advance({ log, apiKey, headers })");
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("a crashed run resumed without its host key is refused before anything is appended", async () => {
  const flow = await loadFlow("id: k\nname: k\nnodes:\n  agent: {kind: agent, prompt: hi}\noutput: ${agent.output}\n");
  const log = freshLog();
  const llmConfig = { provider: "openai", model: "gpt-test", api_key: "sk-test" };
  // Records 0..8 are kept: the agent has started.
  await crashAfter(9, () => runFlow(flow, {}, { log, env: NO_ENV, llmConfig, numWorkers: 0, models: serving(new ScriptedModel([textReply("ok")])) }));
  const before = readFileSync(log);
  expect(await rejected(advance({ log, env: NO_ENV, models: serving(new ScriptedModel([])) }))).toBeInstanceOf(RangeError);
  expect(readFileSync(log).equals(before)).toBe(true);
  const resumed = await advance({ log, env: NO_ENV, apiKey: "sk-test", models: serving(new ScriptedModel([textReply("ok")])) });
  expect([resumed.status, resumed.output]).toEqual(["succeeded", "ok"]);
});
