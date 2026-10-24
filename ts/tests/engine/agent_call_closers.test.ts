/**
 * TS-only: an agent's pending calls across crashes and deadlines. A crash or stop closes the
 * retried agent's open calls by record (`run_resumed.closed_calls`); the replay writes the closer
 * the record names; an `ask_user` timeout closes the call `timed_out` and the model reads 'timed
 * out: no answer'. A timed-out call closes at its own deadline while the reply's other calls stay
 * open, and the agent waits again on those.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { CallClosed, type RunEvent } from "../../src/events.ts";
import { type Response, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ASK_TIMED_OUT_RESULT, closerResult, TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import { type Scheduler, virtualScheduler } from "../../src/platform.ts";
import { advance, complete, pendingCalls, runFlow } from "../../src/run.ts";
import { decodeRecord, NodeStepRecord, type RecordEnvelope } from "../../src/runtime/log.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import { gateTool } from "./inbox_fixtures.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { callReply, crashAfter, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-closers-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);
const START = Date.UTC(2026, 9, 24, 10, 0, 0, 0);
const MINUTE = 60_000;
const CALL = "r/agent/1/0";
const CONFIG = { provider: "openai", model: "test-model" };

const FLOW = `
id: closers
name: closers
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [ask_user, lookup]
output: \${agent.output}
`;

/** The log's records. */
const records = (log: string): RecordEnvelope[] =>
  readFileSync(log, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => decodeRecord(line));

/** A record's kind, a step as its own kind. */
const kindOf = ({ kind, record }: RecordEnvelope) => (record instanceof NodeStepRecord ? String(record.step["kind"]) : kind);

/** The log's record kinds, a step as its own kind. */
const kinds = (log: string) => records(log).map(kindOf);

/** The seq of the first record whose kind (a step: its step kind) is `kind`. */
const seqOf = (log: string, kind: string) => records(log).find((entry) => kindOf(entry) === kind)!.seq;

/** The `closed` words of the log's closer steps, in order. */
const closers = (log: string) =>
  records(log).flatMap(({ record }) =>
    record instanceof NodeStepRecord && typeof record.step["closed"] === "string" ? [record.step["closed"]] : [],
  );

/** The tool results the model read in its request `index`: `[call id, content, is error]`. */
const sentResults = (model: ScriptedModel, index: number) =>
  model.requests[index]!.messages.flatMap((message) => (message.role === "tool" ? [message.parts[0] as ToolResultPart] : [])).map(
    (part) => [part.callId, part.content, part.isError],
  );

/** The reason of each `CallClosed` among `events`: `[call id, reason]`. */
const closedEvents = (events: readonly RunEvent[]) =>
  events.flatMap((event) => (event instanceof CallClosed ? [[event.callId, event.reason]] : []));

/** A scripted model answering `replies` in turn. */
const scripted = (...replies: Response[]) => new ScriptedModel(replies, CONFIG);

/** A host tool `name` answering `answer()`. */
const hostTool = (name: string, answer: () => unknown) =>
  tool({ name, description: `${name}.`, parameters: { type: "object", properties: {} }, execute: answer });

/**
 * Options for `runFlow` over `m`: `ask` is the ask handler, `lookup` answers `found`, the clock
 * starts at `at`; `extra` adds tools. `advance` takes the same minus `runId` (`resumeOf`).
 */
function options(
  log: string,
  m: ScriptedModel,
  ask: (request: { deadline: string | null }) => unknown,
  at: number | Scheduler = START,
  events?: RunEvent[],
  extra: Record<string, ToolEntry> = {},
) {
  return {
    env: hostEnv(dir),
    log,
    runId: "r",
    models: serving(m),
    scheduler: typeof at === "number" ? virtualScheduler(at) : at,
    tools: { ask: ask as ToolEntry, lookup: hostTool("lookup", () => "found"), ...extra },
    onEvent: events === undefined ? null : (event: RunEvent) => void events.push(event),
  };
}

/** `options` for `advance`: the same, without `runId`. */
function resumeOf(...args: Parameters<typeof options>) {
  const { runId: _runId, ...rest } = options(...args);
  return rest;
}

test("each closing reason has its closer text", () => {
  expect(closerResult("crashed")).toEqual({ output: TOOL_OUTCOME_UNKNOWN, closed: "crashed" });
  expect(closerResult("timed_out")).toEqual({ output: "timed out: no answer", closed: "timed_out" });
  expect(ASK_TIMED_OUT_RESULT).toBe("timed out: no answer");
});

test("a crash after an agent's pending step: the resume closes the call, the closer says crashed", async () => {
  let asked = 0;
  const replies = () => scripted(callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("checked"));
  // An uncut run tells where the pending step lands.
  const probe = freshLog();
  await runFlow(await loadFlow(FLOW), {}, options(probe, replies(), () => pending()));
  const log = freshLog();
  await crashAfter(seqOf(probe, "pending") + 1, async () =>
    runFlow(await loadFlow(FLOW), {}, options(log, replies(), () => (asked++, pending()))),
  );
  const events: RunEvent[] = [];
  const m = scripted(textReply("checked"));
  const result = await advance(resumeOf(log, m, () => (asked++, pending()), START + MINUTE, events));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(asked).toBe(1);
  const resumed = records(log).find(({ kind }) => kind === "run_resumed")!;
  expect((resumed.record as { closedCalls: string[] }).closedCalls).toEqual([CALL]);
  expect(closedEvents(events)).toEqual([[CALL, "crashed"]]);
  expect(sentResults(m, 0)).toEqual([["c1", TOOL_OUTCOME_UNKNOWN, true]]);
  expect(closers(log)).toEqual(["crashed"]);
});

/** A crashed log whose last record is the agent's `pending` step for `CALL` (replies `replies()`). */
async function crashedAtPending(replies: () => ScriptedModel, extra: Record<string, ToolEntry> = {}, onAsk = () => {}) {
  const probe = freshLog();
  await runFlow(await loadFlow(FLOW_WITH(extra)), {}, options(probe, replies(), () => pending(), START, undefined, extra));
  const log = freshLog();
  const cut = records(probe).findLastIndex((entry) => kindOf(entry) === "pending") + 1;
  await crashAfter(cut, async () =>
    runFlow(await loadFlow(FLOW_WITH(extra)), {}, options(log, replies(), () => (onAsk(), pending()), START, undefined, extra)),
  );
  expect(kinds(log).at(-1)).toBe("pending");
  return log;
}

/** `FLOW` granting the extra tools too. */
const FLOW_WITH = (extra: Record<string, ToolEntry>) =>
  FLOW.replace("tools: [ask_user, lookup]", `tools: [${["ask_user", "lookup", ...Object.keys(extra)].join(", ")}]`);

test("a complete after run_resumed is refused; the closer follows", async () => {
  let asked = 0;
  const replies = () => scripted(callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("checked"));
  const log = await crashedAtPending(replies, {}, () => asked++);
  // Cut the resume right after its run_resumed.
  await crashAfter(records(log).length + 1, () => advance(resumeOf(log, scripted(), () => pending())));
  expect(kinds(log).at(-1)).toBe("run_resumed");

  const refused = await complete(log, CALL, { output: "yes" }).catch((error: unknown) => error);
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect((refused as DeliveryRefused).reason).toBe("call_settled");
  expect(pendingCalls(log)).toEqual([]);
  const m = scripted(textReply("checked"));
  const result = await advance(resumeOf(log, m, () => (asked++, pending())));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(asked).toBe(1);
  expect(closers(log)).toEqual(["crashed"]);
  expect(sentResults(m, 0)).toEqual([["c1", TOOL_OUTCOME_UNKNOWN, true]]);
});

test("a complete appended while the run is crashed, before run_resumed, is kept", async () => {
  const replies = () => scripted(callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("checked"));
  const log = await crashedAtPending(replies);
  expect(await complete(log, CALL, { output: "yes" })).toBeNumber();
  const m = scripted(textReply("checked"));
  const result = await advance(resumeOf(log, m, () => pending()));

  expect(result.status, result.failure?.message).toBe("succeeded");
  const resumed = records(log).find(({ kind }) => kind === "run_resumed")!;
  expect((resumed.record as { closedCalls: string[] }).closedCalls).toEqual([]);
  expect(sentResults(m, 0)).toEqual([["c1", "yes", false]]);
  expect(closers(log)).toEqual([]);
});

test("a crash while the agent waits closes nothing", async () => {
  const replies = () => scripted(callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("checked"));
  const probe = freshLog();
  await runFlow(await loadFlow(FLOW), {}, options(probe, replies(), () => pending()));
  const log = freshLog();
  await crashAfter(seqOf(probe, "node_waiting") + 1, async () =>
    runFlow(await loadFlow(FLOW), {}, options(log, replies(), () => pending())),
  );
  expect(kinds(log).at(-1)).toBe("node_waiting");

  expect((await advance(resumeOf(log, scripted(), () => pending()))).status).toBe("waiting");
  expect(pendingCalls(log).map((call) => call.callId)).toEqual([CALL]);
  await complete(log, CALL, { output: "yes" });
  const m = scripted(textReply("checked"));
  const result = await advance(resumeOf(log, m, () => pending()));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(sentResults(m, 0)).toEqual([["c1", "yes", false]]);
  expect(closers(log)).toEqual([]);
});

test("a stop with an agent call open closes it crashed", async () => {
  const gate = gateTool();
  const controller = new AbortController();
  const replies = () => scripted(callReply(["ask_user", { question: "Ship?" }, "c1"], ["gate", {}, "c2"]), textReply("checked"));
  const log = freshLog();
  const running = runFlow(await loadFlow(FLOW_WITH({ gate: gate.tool })), {}, {
    ...options(log, replies(), () => pending(), START, undefined, { gate: gate.tool }),
    signal: controller.signal,
  });
  await gate.entered;
  controller.abort(new Error("host stop"));
  await running.catch(() => undefined);

  const m = scripted(textReply("checked"));
  const result = await advance(resumeOf(log, m, () => pending(), START, undefined, { gate: gateTool().tool }));
  expect(result.status, result.failure?.message).toBe("succeeded");
  const resumed = records(log).find(({ kind }) => kind === "run_resumed")!;
  expect((resumed.record as { closedCalls: string[] }).closedCalls).toEqual([CALL]);
  // The ask closed by record and the gate a stop cut: both read as outcome unknown.
  expect(closers(log)).toEqual(["crashed", "crashed"]);
  expect(sentResults(m, 0)).toEqual([
    ["c1", TOOL_OUTCOME_UNKNOWN, true],
    ["c2", TOOL_OUTCOME_UNKNOWN, true],
  ]);
});

test("two pended calls and a crash before node_waiting: both close, in call order", async () => {
  const hold = { hold: hostTool("hold", () => pending()) };
  const replies = () => scripted(callReply(["ask_user", { question: "Ship?" }, "c1"], ["hold", {}, "c2"]), textReply("checked"));
  const log = await crashedAtPending(replies, hold);
  const m = scripted(textReply("checked"));
  const result = await advance(resumeOf(log, m, () => pending(), START, undefined, hold));

  expect(result.status, result.failure?.message).toBe("succeeded");
  const resumed = records(log).find(({ kind }) => kind === "run_resumed")!;
  expect((resumed.record as { closedCalls: string[] }).closedCalls).toEqual([CALL, "r/agent/1/1"]);
  expect(closers(log)).toEqual(["crashed", "crashed"]);
  expect(sentResults(m, 0)).toEqual([
    ["c1", TOOL_OUTCOME_UNKNOWN, true],
    ["c2", TOOL_OUTCOME_UNKNOWN, true],
  ]);
});
