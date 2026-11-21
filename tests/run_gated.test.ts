/**
 * A run a host gates: `runFlow` and `advance` with a `beforeToolCall` hook record it, each asks its
 * own hook about its agents' tool calls, and an unfinished gated run is refused to a resume without
 * one.
 */

import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ComposedFlow } from "../src/compose/build/pipeline.ts";
import { loadFlow } from "../src/compose/loader.ts";
import type { RunEvent } from "../src/events.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { ToolNode } from "../src/nodes/tool/node.ts";
import { advance, cancel, complete, inspectRun, interrupt, pendingCalls, runFlow } from "../src/run.ts";
import { NodeStepRecord, readRunLog, RunResumedRecord, RunStartedRecord } from "../src/runtime/log.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool, tool } from "../src/tools/function_tools.ts";
import type { BeforeToolCall, GatedCall, ToolSource } from "../src/tools/gate.ts";
import { YianRangeError, YianTypeError } from "../src/typesys/errors.ts";
import { gated } from "./helpers/gate.ts";
import { completeByHand } from "./helpers/inbox_fixtures.ts";
import { callReply, crashAfter, graph, NO_ENV, rejected, serving, textReply } from "./helpers/run_state_fixtures.ts";

const ROOT = mkdtempSync(join(tmpdir(), "run-gated-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

/** A hook that runs every call; this file's flows make no agent call, so it is never asked. */
const HOOK: BeforeToolCall = () => undefined;

/** The host tool `lookup`, whose handler returns `pending()`. */
const LOOKUP = new FunctionTool({
  name: "lookup",
  description: "looks something up later",
  parametersJsonSchema: { type: "object", properties: {} },
  execute: () => pending(),
});

/** START -> TOOL node `first` -> TOOL node `second` -> END, each calling `lookup`. */
function twoCalls(): ComposedFlow {
  const nodes = [new ToolNode("first", { toolId: "lookup" }), new ToolNode("second", { toolId: "lookup" })];
  return new ComposedFlow(graph(nodes, [["__start__", "first"], ["first", "second"], ["second", "__end__"]]));
}

/** A run of `twoCalls()` waiting on `first`, started with `hook` when given. */
async function waitingRun(hook: BeforeToolCall | null = null) {
  const log = freshLog();
  const flow = twoCalls();
  const tools = { lookup: LOOKUP };
  const result = await runFlow(flow, {}, gated({ log, env: NO_ENV, tools, numWorkers: 0 }, hook));
  expect(result.status).toBe("waiting");
  return { log, flow, tools, runId: result.runId };
}

/** The `run_resumed` records of the log at `log`. */
const resumes = (log: string) =>
  readRunLog(log).flatMap((envelope) => (envelope.record instanceof RunResumedRecord ? [envelope.record] : []));

/** The message refusing a resume of the gated run at `log` without a hook. */
const refusal = (log: string) =>
  `run log ${log} is gated: a host checked its tool calls with beforeToolCall; ` +
  "resume it with one (advance({ log, beforeToolCall }))";

test("a run started with a hook records gated: true; one without leaves it out", async () => {
  const withHook = await waitingRun(HOOK);
  const without = await waitingRun();
  const start = (log: string) => readRunLog(log)[0]!;
  expect([(start(withHook.log).record as RunStartedRecord).gated, start(withHook.log).payload["gated"]]).toEqual([true, true]);
  expect(Object.hasOwn(start(without.log).payload, "gated")).toBe(false);
});

test("a hook that isn't a function is refused before the log is created or changed", async () => {
  const log = freshLog();
  const bad = 5 as unknown as BeforeToolCall;
  const error = new YianTypeError("beforeToolCall must be a function or null");
  const flow = twoCalls();
  expect(await rejected(runFlow(flow, {}, gated({ log, env: NO_ENV, tools: { lookup: LOOKUP } }, bad)))).toEqual(error);
  expect(existsSync(log)).toBe(false);
  const waiting = await waitingRun();
  const before = readFileSync(waiting.log);
  // Another worker count would write run_resumed.
  const options = { log: waiting.log, flow, tools: waiting.tools, env: NO_ENV, numWorkers: 2 };
  expect(await rejected(advance(gated(options, bad)))).toEqual(error);
  expect(readFileSync(waiting.log).equals(before)).toBe(true);
});

test("an unfinished gated run is refused to an advance without a hook, with the log unchanged", async () => {
  const { log, flow, tools, runId } = await waitingRun(HOOK);
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  const before = readFileSync(log);
  const error = await rejected(advance({ log, flow, tools, env: NO_ENV }));
  expect(error).toEqual(new YianRangeError(refusal(log)));
  expect(readFileSync(log).equals(before)).toBe(true);
  // A crashed gated run is refused the same way.
  const crashed = freshLog();
  await crashAfter(8, () => runFlow(flow, {}, gated({ log: crashed, env: NO_ENV, tools, numWorkers: 0 }, HOOK)));
  expect(await rejected(advance({ log: crashed, flow, tools, env: NO_ENV }))).toEqual(new YianRangeError(refusal(crashed)));
});

test("a gated run's every resume records gated, and its log reads without a hook", async () => {
  const { log, flow, tools, runId } = await waitingRun(HOOK);
  expect(pendingCalls(log).map((call) => call.callId)).toEqual([`${runId}/first`]);
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  // Another worker count forces a resume.
  expect((await advance(gated({ log, flow, tools, env: NO_ENV, numWorkers: 1 }, HOOK))).status).toBe("waiting");
  expect(resumes(log).map((record) => record.gated)).toEqual([true]);
  completeByHand(log, flow, `${runId}/second`, { output: "two" });
  expect((await advance(gated({ log, flow, tools, env: NO_ENV }, HOOK))).status).toBe("succeeded");
  // A finished run returns its result to any advance.
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("succeeded");
  expect(inspectRun(log, { flow }).status).toBe("succeeded");
});

test("a hooked advance of an ungated waiting run records gated for good, even from a clean wait", async () => {
  const { log, flow, tools, runId } = await waitingRun();
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  expect(readRunLog(log).at(-2)!.kind).toBe("run_waiting");
  expect((await advance(gated({ log, flow, tools, env: NO_ENV }, HOOK))).status).toBe("waiting");
  expect(resumes(log).map((record) => [record.retryNodeIds, record.gated])).toEqual([[[], true]]);
  completeByHand(log, flow, `${runId}/second`, { output: "two" });
  expect(await rejected(advance({ log, flow, tools, env: NO_ENV }))).toEqual(new YianRangeError(refusal(log)));
});

test("a hooked advance that can't move appends nothing, and the run stays ungated", async () => {
  const { log, flow, tools, runId } = await waitingRun();
  const before = readFileSync(log);
  expect((await advance(gated({ log, flow, tools, env: NO_ENV }, HOOK))).status).toBe("waiting");
  expect(readFileSync(log).equals(before)).toBe(true);
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("waiting");
  expect(resumes(log)).toEqual([]);
});

test("runFlow and advance each ask their own hook; a pending() tool's note lands on its pending step", async () => {
  const flow = await loadFlow(`
id: gated_agent
name: gated_agent
llm_config: {provider: openai, model: test-model}
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [lookup, find]
output: \${reply.output}
`);
  const find = tool({ name: "find", description: "Find.", parameters: { type: "object", properties: {} }, execute: () => "found" });
  const tools = { lookup: LOOKUP, find };
  const model = new ScriptedModel([callReply(["lookup", {}, "c1"]), callReply(["find", {}, "c2"]), textReply("done")]);
  /** A hook noting each tool it is asked about in `asked`. */
  const noting = (asked: string[], note?: string): BeforeToolCall => (call: GatedCall) => {
    asked.push(call.tool);
    return note === undefined ? undefined : { run: true, note };
  };
  const first: string[] = [];
  const second: string[] = [];
  const log = freshLog();
  const options = { log, env: NO_ENV, tools, models: serving(model) };
  const started = await runFlow(flow, {}, gated(options, noting(first, "a person checks it")));
  expect(started.status, started.failure?.message).toBe("waiting");
  const held = readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord && record.step["kind"] === "pending" ? [record.step] : []));
  expect(held.map((step) => step["note"])).toEqual(["a person checks it"]);
  await complete(log, pendingCalls(log)[0]!.callId, { output: "looked up" }, { env: NO_ENV });
  const done = await advance(gated(options, noting(second)));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([first, second]).toEqual([["lookup"], ["find"]]);
});

describe("a held call through the host calls", () => {
  /** An agent `reply` reading steer inbox `m`, granted `lookup`, whose handler counts its runs in `ran`. */
  const FLOW = `
id: held_agent
name: held_agent
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [lookup]
    steer_inbox: \${input.m}
    output: !Optional(str)
output: \${reply.output}
`;
  const SOURCE: ToolSource = { kind: "host", group: null, tool: "lookup" };

  /**
   * Rewrite the log's pending step of `callId` into a hold, as the gate writes one; the agent
   * loop writes none yet.
   */
  function holdIn(log: string, callId: string): void {
    const lines = readFileSync(log, "utf8").split("\n").filter(Boolean).map((line) => {
      const json = JSON.parse(line);
      const step = json.kind === "node_step" ? json.payload.step : null;
      if (step?.kind !== "pending" || step.call_id !== callId) return line;
      Object.assign(step, { held: true, source: SOURCE, group_end: step.index });
      return JSON.stringify(json);
    });
    writeFileSync(log, `${lines.join("\n")}\n`);
  }

  /** A run waiting on its agent's one `lookup` call, rewritten into a hold; `ran` counts the tool's runs. */
  async function heldRun() {
    const ran: string[] = [];
    const lookup = tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: () => (ran.push("lookup"), pending()) });
    const log = freshLog();
    const options = { log, env: NO_ENV, tools: { lookup }, models: serving(new ScriptedModel([callReply(["lookup", {}, "c1"])])) };
    const result = await runFlow(await loadFlow(FLOW), {}, options);
    expect(result.status, result.failure?.message).toBe("waiting");
    const [call] = pendingCalls(log);
    holdIn(log, call!.callId);
    ran.length = 0;
    return { log, options, ran, callId: call!.callId };
  }

  test("interrupt() and cancel() return the hold held, with a frozen copy of its source", async () => {
    const interrupted = await heldRun();
    const { calls } = await interrupt(interrupted.log, "m", { env: NO_ENV });
    const cancelled = await heldRun();
    const closed = await cancel(cancelled.log, { env: NO_ENV });
    for (const [call] of [calls, closed]) {
      expect([call!.held, call!.source, Object.isFrozen(call!.source)]).toEqual([true, SOURCE, true]);
    }
  });

  test("a resume after a crash before node_waiting keeps the hold, runs no tool and publishes no CallHeld", async () => {
    const { log, options, ran, callId } = await heldRun();
    const lines = readFileSync(log, "utf8").split("\n").filter(Boolean);
    writeFileSync(log, `${lines.slice(0, lines.findIndex((line) => line.includes('"held":true')) + 1).join("\n")}\n`);
    const events: RunEvent[] = [];
    const resumed = await advance({ ...options, onEvent: (event) => void events.push(event) });
    expect(resumed.status, resumed.failure?.message).toBe("waiting");
    expect(events.map((event) => event.type)).not.toContain("CallHeld");
    expect(events.map((event) => event.type)).not.toContain("CallClosed");
    expect(ran).toEqual([]);
    expect(pendingCalls(log).map((call) => [call.callId, call.held])).toEqual([[callId, true]]);
  });
});
